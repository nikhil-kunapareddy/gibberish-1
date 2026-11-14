#!/usr/bin/env python3
"""Stage 3: instruction-tune a pretrained checkpoint (loss masked to completions).

    python scripts/sft.py --sft configs/sft/smoke.yaml
    python scripts/sft.py --sft.init_from=runs/pretrain-smoke/ckpt_last.pt
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from tinystories_lm.posttrain.sft import run_sft
from tinystories_lm.utils.config import build_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft", default="configs/sft/smoke.yaml")
    args, overrides = parser.parse_known_args()

    cfg = build_config({"sft": args.sft}, overrides)
    save_config(cfg, f"{cfg.sft.out_dir}/config.yaml")
    result = run_sft(cfg.sft, full_config=cfg.to_dict())
    print(
        f"[sft] done in {result.elapsed_sec:.1f}s: train_loss={result.final_train_loss:.4f} "
        f"best_val_loss={result.best_val_loss:.4f} -> {result.best_ckpt}"
    )


if __name__ == "__main__":
    main()
