"""Shared SAE / transcoder loading helpers for Phase 5 (Stages 01 + 02).

Two load paths:

  * **registry**  — the (release, sae_id) is in the sae-lens pretrained
    directory; load via `SAE.from_pretrained` (transcoders fall back to
    `Transcoder.from_pretrained`).
  * **hf_disk**   — a Hugging Face repo that stores SAEs in sae-lens *training
    output* format (a folder with `cfg.json` + `sae_weights.safetensors`
    [+ optional `sparsity.safetensors`]) but is NOT registered in the
    directory. Download the folder's files and load via `SAE.load_from_disk`.
    Used for `Geaming/Llama-3.2-3B-Instruct_SAEs`, the only residual SAE found
    for Llama-3.2-3B-Instruct.

The sae-lens classes are passed in by the caller so this module never imports
`sae_lens` itself (the stages keep the optional-dependency guard).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

DISK_FILES = ("cfg.json", "sae_weights.safetensors", "sparsity.safetensors")


def _unwrap(res):
    """sae-lens >=6 returns a bare module; older returns (sae, cfg, sparsity)."""
    return res[0] if isinstance(res, tuple) else res


def load_registry_module(SAE, Transcoder, release: str, sae_id: str, device: str, kind: str):
    """Load a registered SAE/transcoder by (release, sae_id)."""
    try:
        return _unwrap(SAE.from_pretrained(release=release, sae_id=sae_id, device=device))
    except Exception as e_sae:
        if kind == "transcoder" and Transcoder is not None:
            log.info("SAE.from_pretrained rejected the transcoder (%s); trying Transcoder.", e_sae)
            return _unwrap(
                Transcoder.from_pretrained(release=release, sae_id=sae_id, device=device)
            )
        raise


def load_hf_disk_module(
    SAE,
    hf_repo: str,
    subdir: str,
    device: str,
    files: tuple[str, ...] = DISK_FILES,
):
    """Download a sae-lens-format folder from an HF repo and load it.

    `sparsity.safetensors` is treated as optional (some repos omit it). Returns
    the loaded SAE.
    """
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    local_dir: Path | None = None
    for fn in files:
        remote = f"{subdir}/{fn}" if subdir else fn
        try:
            path = hf_hub_download(hf_repo, remote, token=token)
        except Exception as e:
            if fn == "sparsity.safetensors":
                log.info("No sparsity.safetensors in %s/%s — continuing without it.",
                         hf_repo, subdir)
                continue
            raise RuntimeError(f"Failed to fetch {hf_repo}/{remote}: {e}") from e
        local_dir = Path(path).parent

    if local_dir is None:
        raise RuntimeError(f"Could not download any SAE files from {hf_repo}/{subdir}")
    log.info("Loading SAE from disk: %s", local_dir)
    return SAE.load_from_disk(str(local_dir), device=device)


def load_qwen_scope_module(
    hf_repo: str,
    layer: int,
    k: int,
    device: str,
    *,
    file_template: str = "layer{layer}.sae.pt",
):
    """Load a Qwen-Scope residual TopK-SAE checkpoint (`.pt` dict) for one layer.

    Qwen-Scope SAEs are NOT sae-lens repos: each `layer{n}.sae.pt` is a plain
    dict of four tensors — W_enc (d_sae,d_in), W_dec (d_in,d_sae), b_enc (d_sae,),
    b_dec (d_in,) — and encoding is TopK. Verified against the official model card
    (Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50, 2026-06-21). Returns a small nn.Module
    exposing `.encode(x) -> [B, d_sae]`, `.d_sae`, `.d_in`, `.k`, `.hook_name`,
    so Phase 5 Stages 01/02 treat it exactly like an sae-lens module.
    """
    import torch
    from torch import nn
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    fname = file_template.format(layer=layer)
    path = hf_hub_download(hf_repo, fname, token=token)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("W_enc", "W_dec", "b_enc", "b_dec"):
        if key not in sd:
            raise RuntimeError(f"{hf_repo}/{fname} missing key {key!r}; got {list(sd)}")

    class QwenScopeTopKSAE(nn.Module):
        """TopK SAE: acts = scatter(topk(x @ W_enc.T + b_enc, k)). Matches the
        official Qwen-Scope `get_feature_acts` (no b_dec subtraction, no ReLU —
        TopK keeps the k largest pre-activations)."""

        def __init__(self, state: dict, k: int):
            super().__init__()
            # requires_grad=False params so .parameters() is non-empty (Stage 02
            # reads device/dtype via next(sae.parameters())) but nothing trains.
            self.W_enc = nn.Parameter(state["W_enc"], requires_grad=False)  # (d_sae, d_in)
            self.W_dec = nn.Parameter(state["W_dec"], requires_grad=False)  # (d_in, d_sae)
            self.b_enc = nn.Parameter(state["b_enc"], requires_grad=False)  # (d_sae,)
            self.b_dec = nn.Parameter(state["b_dec"], requires_grad=False)  # (d_in,)
            self.d_sae = int(self.W_enc.shape[0])
            self.d_in = int(self.W_enc.shape[1])
            self.k = int(k)
            self.hook_name = "resid_post"

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            pre = x @ self.W_enc.T + self.b_enc            # [B, d_sae]
            vals, idx = pre.topk(self.k, dim=-1)
            acts = torch.zeros_like(pre)
            acts.scatter_(-1, idx, vals)
            return acts

        def forward(self, x):  # parity with sae-lens modules
            return self.encode(x)

    sae = QwenScopeTopKSAE(sd, k).to(device)
    sae.eval()
    log.info("Loaded Qwen-Scope TopK SAE %s/%s (d_sae=%d, d_in=%d, k=%d)",
             hf_repo, fname, sae.d_sae, sae.d_in, sae.k)
    return sae
