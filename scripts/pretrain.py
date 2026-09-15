#!/usr/bin/env python3
"""Stage 2: pretrain a model on the prepared token stream.

    python scripts/pretrain.py --pretrain configs/pretrain/smoke.yaml --model configs/model/smoke-1m.yaml
    python scripts/pretrain.py --pretrain.lr=3e-4 --pretrain.max_steps=1000
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from tinystories_lm.pretrain.train import run_pretrain
from tinystories_lm.utils.config import build_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrain", default="configs/pretrain/smoke.yaml")
    parser.add_argument("--model", default="configs/model/smoke-1m.yaml")
    args, overrides = parser.parse_known_args()

    cfg = build_config({"pretrain": args.pretrain, "model": args.model}, overrides)
    save_config(cfg, f"{cfg.pretrain.out_dir}/config.yaml")
    result = run_pretrain(cfg.pretrain, cfg.model, full_config=cfg.to_dict())
    print(
        f"[pretrain] done in {result.elapsed_sec:.1f}s: "
        f"train_loss={result.final_train_loss:.4f} best_val_loss={result.best_val_loss:.4f} "
        f"-> {result.best_ckpt}"
    )


if __name__ == "__main__":
    main()
