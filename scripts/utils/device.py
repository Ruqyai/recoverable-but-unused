"""Hardware-aware device detection and per-device tuning.

Priority order (best → worst for our workload):

    1. CUDA (NVIDIA GPU)        — fastest, supports device_map="auto"
    2. XLA  (TPU via torch_xla) — fast on TPU pods; requires torch_xla install
    3. MPS  (Apple Silicon)     — M1/M2/M3/M4; tuned per chip when detectable
    4. CPU                      — fallback only; very slow for 4B+ models

`detect_device()` returns a `DeviceConfig` with the device string AND
hardware-tuned recommendations (dtype, batch_size_multiplier, cache
management hooks, whether device_map="auto" is supported, etc.). The
return value is meant to be applied at the start of any extraction or
generation script.

Apple Silicon chip detection (Darwin only) reads
`sysctl -n machdep.cpu.brand_string`, which returns strings like
"Apple M2", "Apple M2 Pro", "Apple M3 Max". The chip name is surfaced in
`DeviceConfig.chip_name` and recorded in run_metadata.json so reviewers
can verify which device produced a given activation tensor.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeviceConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeviceConfig:
    """Resolved device + per-device tuning hints.

    Attributes:
        device:          PyTorch device string ("cuda", "mps", "xla", "cpu").
        backend:         Finer-grained label ("cuda", "xla", "mps_m1",
                         "mps_m2", "mps_m2_pro", "mps_m3", "mps_other",
                         "cpu").
        dtype:           Recommended floating-point dtype for model weights.
        batch_size_multiplier:
                         Multiply the model's default BATCH_SIZE by this to
                         get a device-appropriate batch size. >1 = bigger
                         batches than the per-model default; <1 = smaller.
        use_device_map_auto:
                         Whether to pass device_map="auto" to
                         AutoModelForCausalLM.from_pretrained.
        empty_cache:     Callable that clears the device cache between
                         batches (no-op for CPU; torch.cuda.empty_cache /
                         torch.mps.empty_cache / xm.mark_step otherwise).
        chip_name:       Apple Silicon chip name when device == "mps" and
                         detection succeeded ("Apple M2", "Apple M2 Pro",
                         ...); None otherwise.
        total_memory_gb: Total system memory in GB when knowable.
        notes:           Human-readable summary surfaced in logs and
                         run_metadata.json.
    """
    device: str
    backend: str
    dtype: torch.dtype
    batch_size_multiplier: float
    use_device_map_auto: bool
    empty_cache: Callable[[], None]
    chip_name: str | None = None
    total_memory_gb: float | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        """JSON-serializable summary for run_metadata.json."""
        return {
            "device": self.device,
            "backend": self.backend,
            "dtype": str(self.dtype),
            "batch_size_multiplier": self.batch_size_multiplier,
            "use_device_map_auto": self.use_device_map_auto,
            "chip_name": self.chip_name,
            "total_memory_gb": self.total_memory_gb,
            "notes": self.notes,
        }

    def apply_batch_size(self, base: int, minimum: int = 1) -> int:
        """Apply the multiplier to a base batch size."""
        return max(minimum, int(round(base * self.batch_size_multiplier)))


# ---------------------------------------------------------------------------
# Apple Silicon chip detection
# ---------------------------------------------------------------------------

def _detect_apple_chip() -> str | None:
    """Return the Apple Silicon chip name (e.g. 'Apple M2'), or None.

    Uses `sysctl machdep.cpu.brand_string`. Only valid on macOS arm64.
    """
    if platform.system() != "Darwin":
        return None
    if platform.machine() != "arm64":
        return None
    sysctl = shutil.which("sysctl")
    if sysctl is None:
        return None
    try:
        out = subprocess.check_output(
            [sysctl, "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _macos_total_memory_gb() -> float | None:
    """Total physical memory on macOS, in GiB."""
    if platform.system() != "Darwin":
        return None
    sysctl = shutil.which("sysctl")
    if sysctl is None:
        return None
    try:
        out = subprocess.check_output(
            [sysctl, "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        bytes_ = int(out.decode().strip())
        return round(bytes_ / (1024 ** 3), 1)
    except Exception:
        return None


def _classify_apple_chip(chip_name: str) -> str:
    """Map an Apple Silicon brand string to a backend label."""
    s = chip_name.lower()
    if "m1" in s:
        return "mps_m1"
    if "m2" in s:
        return "mps_m2_pro" if ("pro" in s or "max" in s or "ultra" in s) else "mps_m2"
    if "m3" in s:
        return "mps_m3"
    if "m4" in s:
        return "mps_m4"
    return "mps_other"


def _mps_batch_multiplier_and_dtype(backend: str) -> tuple[float, torch.dtype]:
    """Per-chip batch-size multiplier and dtype.

    Multipliers are relative to the per-model BATCH_SIZE set in
    scripts/models/<key>.py, which is sized
    for a baseline M2 (24 GB unified). Larger chips get >1 multiplier;
    smaller chips get <1.

    All Apple Silicon uses float16 — MPS supports it on residual-stream
    extraction and generation. bfloat16 is partially supported on M2+ but
    fp16 is the safer default for hidden-state extraction.
    """
    if backend == "mps_m1":
        return 0.75, torch.float16     # 8-16 GB unified; conservative
    if backend == "mps_m2":
        return 1.00, torch.float16     # baseline (~16-24 GB unified)
    if backend == "mps_m2_pro":
        return 1.50, torch.float16     # M2 Pro/Max/Ultra (~32-96 GB)
    if backend == "mps_m3":
        return 1.25, torch.float16     # M3 base; better than M2 baseline
    if backend == "mps_m4":
        return 1.50, torch.float16
    return 1.00, torch.float16          # unknown Apple chip


# ---------------------------------------------------------------------------
# TPU / XLA detection
# ---------------------------------------------------------------------------

def _detect_xla() -> object | None:
    """Return the torch_xla.core.xla_model module if a TPU is available."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
    except ImportError:
        return None
    try:
        _ = xm.xla_device()  # raises if no TPU
        return xm
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cache-management helpers
# ---------------------------------------------------------------------------

def _noop_empty_cache() -> None:
    pass


def _cuda_empty_cache() -> None:
    torch.cuda.empty_cache()


def _mps_empty_cache() -> None:
    # torch.mps.empty_cache exists on PyTorch >= 2.0; guard for older builds.
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _make_xla_empty_cache(xm) -> Callable[[], None]:
    def _empty_cache() -> None:
        xm.mark_step()
    return _empty_cache


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

def detect_device(
    prefer: str | None = None,
) -> DeviceConfig:
    """Pick the best available device and return tuned settings.

    Priority: CUDA > XLA (TPU) > MPS > CPU. `prefer` (one of "cuda",
    "xla", "mps", "cpu") forces a specific backend when available; an
    unavailable preference triggers a warning and falls back to the next
    best.
    """
    # 1) CUDA (NVIDIA GPU)
    if (prefer in (None, "cuda")) and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "cuda"
        total_gb = None
        try:
            total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
        except Exception:
            pass
        # bfloat16 if available (Ampere+); else float16.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return DeviceConfig(
            device="cuda",
            backend="cuda",
            dtype=dtype,
            batch_size_multiplier=2.0,
            use_device_map_auto=True,
            empty_cache=_cuda_empty_cache,
            chip_name=name,
            total_memory_gb=total_gb,
            notes=f"CUDA GPU detected: {name} ({total_gb} GB)" if total_gb else f"CUDA GPU: {name}",
        )

    # 2) XLA (TPU)
    xm = _detect_xla() if prefer in (None, "xla") else None
    if xm is not None:
        return DeviceConfig(
            device="xla",
            backend="xla",
            dtype=torch.bfloat16,            # TPU sweet spot
            batch_size_multiplier=4.0,       # TPUs are happy with big batches
            use_device_map_auto=False,       # XLA handles placement itself
            empty_cache=_make_xla_empty_cache(xm),
            chip_name="TPU",
            total_memory_gb=None,
            notes=(
                "TPU detected via torch_xla. Note: scripts must call "
                "xm.xla_device() for the actual device handle."
            ),
        )

    # 3) MPS (Apple Silicon)
    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if (prefer in (None, "mps")) and mps_available:
        chip = _detect_apple_chip()
        backend = _classify_apple_chip(chip) if chip else "mps_other"
        bs_mult, dtype = _mps_batch_multiplier_and_dtype(backend)
        total_gb = _macos_total_memory_gb()
        notes = (
            f"Apple Silicon MPS detected: {chip} "
            f"({total_gb} GB unified)" if (chip and total_gb)
            else "Apple Silicon MPS detected"
        )
        if backend == "mps_m2":
            notes += " — M2 tuned defaults applied (float16, batch ×1.0)"
        return DeviceConfig(
            device="mps",
            backend=backend,
            dtype=dtype,
            batch_size_multiplier=bs_mult,
            use_device_map_auto=False,        # not supported on MPS
            empty_cache=_mps_empty_cache,
            chip_name=chip,
            total_memory_gb=total_gb,
            notes=notes,
        )

    # 4) CPU fallback
    if prefer is not None and prefer != "cpu":
        log.warning(
            f"Preferred device {prefer!r} not available; falling back to CPU."
        )
    return DeviceConfig(
        device="cpu",
        backend="cpu",
        dtype=torch.float32,
        batch_size_multiplier=0.25,
        use_device_map_auto=False,
        empty_cache=_noop_empty_cache,
        chip_name=None,
        total_memory_gb=None,
        notes="CPU only — extraction will be slow; consider GPU/MPS.",
    )


# ---------------------------------------------------------------------------
# HF model loading helper
# ---------------------------------------------------------------------------

def get_text_base_model(model, *, is_multimodal: bool = False, submodule_path: str | None = None):
    """Return the submodule to call for hidden-state extraction.

    Text models → `model.model` (skips lm_head). Multimodal VLMs → the language
    submodule (Gemma-3: `model.model.language_model` = Gemma3TextModel), resolved
    from `submodule_path` (relative to the full model, default 'model.language_model').
    The language submodule accepts input_ids + output_hidden_states; calling the
    VLM wrapper (`model.model`) instead would expect pixel inputs.
    """
    if is_multimodal or submodule_path:
        path = submodule_path or "model.language_model"
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj
    return getattr(model, "model", model)


def resolve_dtype(name: str | None) -> torch.dtype | None:
    """Map a PREFERRED_DTYPE string ('bfloat16'/'float16'/'float32') to a
    torch.dtype, or None if unset. Used to override the device default for
    models that need a specific precision (e.g. Gemma-3 needs bfloat16 on MPS)."""
    if not name:
        return None
    return {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
    }.get(str(name).lower())


def load_hf_model(
    model_id: str,
    cfg: DeviceConfig,
    *,
    hf_token: str | None = None,
    trust_remote_code: bool = False,
    dtype: torch.dtype | None = None,
    model_class: str | None = None,
):
    """Load an HF model with device-aware kwargs (AutoModelForCausalLM by default).

    Returns the loaded `model` placed on `cfg.device`. The caller is
    responsible for `model.eval()` and tokenizer loading.

    `dtype` overrides the device-recommended `cfg.dtype`. Needed for Gemma-3
    models, whose large residual norms OVERFLOW float16 on MPS (NaN at deeper
    layers) — pass `torch.bfloat16` (same exponent range as float32, no overflow).
    The per-model `PREFERRED_DTYPE` config field drives this.

    `model_class` names the concrete HF class to load with when
    `AutoModelForCausalLM` is wrong — i.e. multimodal VLMs whose checkpoint maps
    to a `*ForConditionalGeneration` class (e.g. "Gemma3ForConditionalGeneration"
    for google/gemma-3-4b-it). Pass the per-model `MODEL_CLASS` config field;
    leave None for the text-only models. The text decoder of the loaded VLM is
    then reached via `get_text_base_model(..., submodule_path=cfg.TEXT_SUBMODULE_PATH)`.
    """
    import transformers

    # Resolve the loader class: explicit MODEL_CLASS for VLMs, else the generic
    # causal-LM auto-class (identical to the previous unconditional behaviour).
    if model_class:
        try:
            loader = getattr(transformers, model_class)
        except AttributeError as e:
            raise ValueError(
                f"MODEL_CLASS={model_class!r} is not exposed by transformers "
                f"{transformers.__version__}; check the model config / upgrade transformers."
            ) from e
    else:
        loader = transformers.AutoModelForCausalLM

    use_dtype = dtype or cfg.dtype
    kwargs: dict = {
        "torch_dtype": use_dtype,
        "token": hf_token,
        "trust_remote_code": trust_remote_code,
    }
    if cfg.use_device_map_auto:
        kwargs["device_map"] = "auto"

    log.info(
        f"Loading {model_id} via {loader.__name__} "
        f"(dtype={use_dtype}, device={cfg.device})"
    )
    model = loader.from_pretrained(model_id, **kwargs)

    if not cfg.use_device_map_auto and cfg.device != "xla":
        model = model.to(cfg.device)
    elif cfg.device == "xla":
        # Move via xla_model
        import torch_xla.core.xla_model as xm  # type: ignore
        model = model.to(xm.xla_device())

    return model


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def log_device_info(cfg: DeviceConfig) -> None:
    """Emit a banner describing the resolved device + tuning."""
    log.info("─" * 70)
    log.info("Device detection")
    log.info(f"  device                : {cfg.device}")
    log.info(f"  backend               : {cfg.backend}")
    log.info(f"  dtype                 : {cfg.dtype}")
    log.info(f"  batch-size multiplier : {cfg.batch_size_multiplier}")
    log.info(f"  device_map='auto'     : {cfg.use_device_map_auto}")
    if cfg.chip_name:
        log.info(f"  chip                  : {cfg.chip_name}")
    if cfg.total_memory_gb is not None:
        log.info(f"  total memory (GB)     : {cfg.total_memory_gb}")
    if cfg.notes:
        log.info(f"  notes                 : {cfg.notes}")
    log.info("─" * 70)
