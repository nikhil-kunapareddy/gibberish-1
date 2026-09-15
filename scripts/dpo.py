#!/usr/bin/env python3
"""Stage 4: DPO against a frozen reference copy of the SFT model.

Validates mechanics only at this scale -- see the README.

    python scripts/dpo.py --dpo configs/dpo/smoke.yaml
    python scripts/dpo.py --dpo.beta=0.5 --dpo.num_pairs=128
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from tinystories_lm.posttrain.dpo import run_dpo
from tinystories_lm.utils.config import build_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dpo", default="configs/dpo/smoke.yaml")
    args, overrides = parser.parse_known_args()

    cfg = build_config({"dpo": args.dpo}, overrides)
    save_config(cfg, f"{cfg.dpo.out_dir}/config.yaml")
    result = run_dpo(cfg.dpo, full_config=cfg.to_dict())
    print(f"[dpo] done in {result.elapsed_sec:.1f}s: loss={result.final_train_loss:.4f} -> {result.last_ckpt}")


if __name__ == "__main__":
    main()
