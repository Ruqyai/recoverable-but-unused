#!/usr/bin/env python3
"""Check chat templates for all candidate models."""

from transformers import AutoTokenizer

MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "google/gemma-2-9b-it",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]

TEST_PROMPT = "Hello, how are you?"


def check_model(model_id: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"MODEL: {model_id}")
    print(f"{'=' * 80}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"FAILED TO LOAD: {type(e).__name__}: {str(e)[:300]}")
        return

    print(f"\nTokenizer class: {type(tokenizer).__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"\nSpecial tokens:")
    print(f"  bos_token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
    print(f"  eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad_token: {tokenizer.pad_token!r}")
    print(f"  unk_token: {tokenizer.unk_token!r}")

    print(f"\nHas chat_template attribute: {hasattr(tokenizer, 'chat_template')}")
    if hasattr(tokenizer, 'chat_template'):
        ct = tokenizer.chat_template
        if ct:
            print(f"Chat template (first 500 chars):")
            print(ct[:500])
        else:
            print("Chat template is empty/None")

    print(f"\nApply chat template test:")
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": TEST_PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"Formatted output (repr):")
        print(repr(formatted))
        print(f"\nFormatted output (display):")
        print(formatted)
        print(f"\nLength: {len(formatted)} chars")

        # Tokenize and check
        tokens = tokenizer(formatted, return_tensors="pt")
        print(f"Tokenized length: {tokens['input_ids'].shape[1]} tokens")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    for model_id in MODELS:
        check_model(model_id)
