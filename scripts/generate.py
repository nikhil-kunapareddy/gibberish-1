#!/usr/bin/env python3
"""Sample from a checkpoint.

    python scripts/generate.py --checkpoint runs/pretrain-smoke/ckpt_best.pt --prompt "Once upon a time"
    python scripts/generate.py --greedy --max-new-tokens 200
    python scripts/generate.py --profile          # tokens/sec with and without the KV cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from tinystories_lm.data.tokenizer import BPETokenizer
from tinystories_lm.inference.generate import generate_text, profile_generation
from tinystories_lm.utils.checkpoint import (
    assert_tokenizer_match,
    load_model_from_checkpoint,
)
from tinystories_lm.utils.config import GenerateConfig
from tinystories_lm.utils.device import resolve_device
from tinystories_lm.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="runs/pretrain-smoke/ckpt_best.pt")
    parser.add_argument("--data-dir", default="data/smoke", help="provides the tokenizer")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true", help="disable the KV cache")
    parser.add_argument("--profile", action="store_true", help="report tokens/sec with and without the cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    # The flags above *are* this stage's config: GenerateConfig holds the
    # defaults and the validation, so there is one definition of each knob.
    cfg = GenerateConfig(
        checkpoint=args.checkpoint, data_dir=args.data_dir, prompt=args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, greedy=args.greedy,
        num_samples=args.num_samples, use_cache=not args.no_cache,
        profile=args.profile, device=args.device, seed=args.seed,
    )
    cfg.validate()

    seed_everything(cfg.seed)
    device_info = resolve_device(cfg.device)
    model, ckpt = load_model_from_checkpoint(cfg.checkpoint, device_info.device)
    tokenizer = BPETokenizer.load(Path(cfg.data_dir))
    assert_tokenizer_match(ckpt, tokenizer.hash(), where=cfg.checkpoint)

    if cfg.profile:
        print(json.dumps(profile_generation(model, tokenizer, cfg.prompt, cfg.max_new_tokens,
                                            device_info=device_info), indent=2))
        return

    for i, gen in enumerate(
        generate_text(
            model, tokenizer, cfg.prompt,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
            top_k=cfg.top_k, top_p=cfg.top_p, greedy=cfg.greedy,
            num_samples=cfg.num_samples, use_cache=cfg.use_cache,
            device_info=device_info,
        )
    ):
        header = f"--- sample {i + 1}/{cfg.num_samples} ({gen.n_new_tokens} tokens, {gen.tokens_per_sec:.0f} tok/s) ---"
        print(f"\n{header}\n{gen.text}")


if __name__ == "__main__":
    main()
