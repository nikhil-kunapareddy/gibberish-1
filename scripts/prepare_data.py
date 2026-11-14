#!/usr/bin/env python3
"""Stage 1: download TinyStories, train the BPE, write .bin + meta.json.

    python scripts/prepare_data.py --data configs/data/smoke.yaml
    python scripts/prepare_data.py --data configs/data/full.yaml --data.num_stories=100000
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (sys.path side effect)

from tinystories_lm.data.prepare import prepare
from tinystories_lm.utils.config import build_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="configs/data/smoke.yaml", help="data config YAML")
    parser.add_argument("--quiet", action="store_true")
    args, overrides = parser.parse_known_args()

    cfg = build_config({"data": args.data}, overrides)
    meta = prepare(cfg.data, quiet=args.quiet)
    save_config(cfg, f"{cfg.data.out_dir}/config.yaml")
    print(
        f"[prepare_data] done: {meta['splits']['train']['n_tokens']:,} train tokens, "
        f"{meta['splits']['val']['n_tokens']:,} val tokens -> {cfg.data.out_dir}"
    )


if __name__ == "__main__":
    main()
