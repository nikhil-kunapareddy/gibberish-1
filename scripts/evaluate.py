#!/usr/bin/env python3
"""Stage 5: validation perplexity + LLM-as-judge scores.

    python scripts/evaluate.py --eval configs/eval/smoke.yaml
    TINYSTORIES_JUDGE=api python scripts/evaluate.py --eval.checkpoint=runs/sft-smoke/ckpt_best.pt
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from tinystories_lm.eval.harness import run_eval
from tinystories_lm.utils.checkpoint import describe
from tinystories_lm.utils.config import build_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", default="configs/eval/smoke.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="also print the checkpoint's metadata")
    args, overrides = parser.parse_known_args()

    cfg = build_config({"eval": args.eval}, overrides)
    if args.verbose:
        print(describe(cfg.eval.checkpoint))
    save_config(cfg, f"{cfg.eval.out_dir}/config.yaml")
    report = run_eval(cfg.eval, full_config=cfg.to_dict())
    print(json.dumps({k: report[k] for k in ("perplexity", "judge")}, indent=2))


if __name__ == "__main__":
    main()
