#!/usr/bin/env python3
"""Run every stage end to end, assert the nine correctness gates, time it all.

    python scripts/smoke_test.py                 # the whole thing (also `make smoke`)
    python scripts/smoke_test.py --skip-gates    # pipeline only
    python scripts/smoke_test.py --device cpu    # force CPU

The generated story at the end will be close to gibberish: the smoke model is
1.5M parameters trained for 200 steps on 5k stories. That is a pass. The gates
are the success criterion, not the sample.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import _bootstrap  # noqa: F401

from tinystories_lm.data.prepare import prepare
from tinystories_lm.data.tokenizer import BPETokenizer
from tinystories_lm.eval.harness import run_eval
from tinystories_lm.inference.generate import generate_text
from tinystories_lm.posttrain.dpo import run_dpo
from tinystories_lm.posttrain.sft import run_sft
from tinystories_lm.pretrain.train import run_pretrain
from tinystories_lm.utils.checkpoint import load_model_from_checkpoint
from tinystories_lm.utils.config import build_config, save_config
from tinystories_lm.utils.device import resolve_device
from tinystories_lm.utils.logging import format_timing_table

REPO_ROOT = Path(__file__).resolve().parents[1]

# The nine non-negotiable gates, keyed by the substring that identifies their
# tests. Every gate must have at least one test, and all of them must pass.
GATES: list[tuple[str, str]] = [
    ("gate1_", "1. Tokenizer roundtrip (encode->decode is exact)"),
    ("gate2_", "2. Off-by-one (y[i,t] == x[i,t+1])"),
    ("gate3_", "3. No split leakage (train/val hashes disjoint)"),
    ("gate4_", "4. Untrained loss within 5% of ln(vocab_size)"),
    ("gate5_", "5. Single-batch overfit (loss < 0.1 in 200 steps)"),
    ("gate6_", "6. Causal mask integrity (no information flows backwards)"),
    ("gate7_", "7. Resume equivalence (20+20 steps == 40 steps)"),
    ("gate8_", "8. KV cache equivalence (cached == uncached decoding)"),
    ("gate9_", "9. SFT mask correctness (prompt tokens contribute no loss)"),
]

# Matches parametrised ids too: `test_gate4_...[8192] PASSED`.
RESULT_RE = re.compile(r"::(test_\w+(?:\[[^\]]*\])?)\s+(PASSED|FAILED|ERROR|SKIPPED)")


@dataclass
class Timing:
    rows: list[tuple[str, float]] = field(default_factory=list)

    def record(self, name: str, seconds: float) -> None:
        self.rows.append((name, seconds))

    def stage(self, name: str):
        timing = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.time()
                print(f"\n{'=' * 72}\n== {name}\n{'=' * 72}", flush=True)
                return self

            def __exit__(self, *exc):
                timing.record(name, time.time() - self.t0)
                return False

        return _Ctx()


def run_gates(timing: Timing, extra_args: list[str]) -> bool:
    """Run the test suite and report each gate's status. Returns True if all pass."""
    with timing.stage("gates (pytest)"):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-v", "--no-header", "-p", "no:cacheprovider", *extra_args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**_env(), "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        results = dict(RESULT_RE.findall(proc.stdout))
        if not results:
            print(proc.stdout[-4000:], file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit("smoke test: pytest produced no results at all")

        all_ok = True
        print(f"\n{'-' * 72}\nCorrectness gates\n{'-' * 72}")
        for key, label in GATES:
            tests = {n: r for n, r in results.items() if key in n}
            failed = [n for n, r in tests.items() if r in ("FAILED", "ERROR")]
            if not tests:
                status, all_ok = "MISSING", False
            elif failed:
                status, all_ok = "FAIL", False
            else:
                status = "PASS"
            print(f"  [{status:>7}] {label}  ({len(tests)} test{'s' if len(tests) != 1 else ''})")
            for name in failed:
                print(f"            failed: {name}")

        other_failed = [
            n for n, r in results.items()
            if r in ("FAILED", "ERROR") and not any(k in n for k, _ in GATES)
        ]
        if other_failed:
            all_ok = False
            print(f"  [{'FAIL':>7}] {len(other_failed)} non-gate test(s) failed: {other_failed[:5]}")
        print(f"  {sum(1 for r in results.values() if r == 'PASSED')}/{len(results)} tests passed")

        if not all_ok:
            print("\n--- pytest output (tail) ---\n" + proc.stdout[-4000:], file=sys.stderr)
        return all_ok


def _env() -> dict:
    import os

    return dict(os.environ)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="configs/data/smoke.yaml")
    parser.add_argument("--model", default="configs/model/smoke-1m.yaml")
    parser.add_argument("--pretrain", default="configs/pretrain/smoke.yaml")
    parser.add_argument("--sft", default="configs/sft/smoke.yaml")
    parser.add_argument("--dpo", default="configs/dpo/smoke.yaml")
    parser.add_argument("--eval", default="configs/eval/smoke.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-gates", action="store_true", help="run the pipeline only")
    parser.add_argument("--gates-only", action="store_true", help="run the gates only")
    args, overrides = parser.parse_known_args()

    t0 = time.time()
    timing = Timing()
    device_override = [f"--pretrain.device={args.device}", f"--sft.device={args.device}",
                       f"--dpo.device={args.device}", f"--eval.device={args.device}"]
    print(f"tinystories-lm smoke test | device={resolve_device(args.device)} | python {sys.version.split()[0]}")

    gates_ok = None if args.skip_gates else run_gates(timing, [])
    if args.gates_only:
        _finish(timing, t0, gates_ok, sample=None)
        return

    # -- stage 1: data ------------------------------------------------------- #
    with timing.stage("prepare_data"):
        cfg = build_config({"data": args.data}, overrides)
        prepare(cfg.data)
        save_config(cfg, f"{cfg.data.out_dir}/config.yaml")

    # -- stage 2: pretrain ---------------------------------------------------- #
    with timing.stage("pretrain"):
        cfg = build_config(
            {"pretrain": args.pretrain, "model": args.model}, overrides + device_override[:1]
        )
        save_config(cfg, f"{cfg.pretrain.out_dir}/config.yaml")
        pre = run_pretrain(cfg.pretrain, cfg.model, full_config=cfg.to_dict())
        print(f"-> {pre.best_ckpt} (val loss {pre.best_val_loss:.4f})")

    # -- stage 3: SFT --------------------------------------------------------- #
    with timing.stage("sft"):
        cfg = build_config({"sft": args.sft}, overrides + device_override[1:2])
        save_config(cfg, f"{cfg.sft.out_dir}/config.yaml")
        sft = run_sft(cfg.sft, full_config=cfg.to_dict())
        print(f"-> {sft.best_ckpt} (val loss {sft.best_val_loss:.4f})")

    # -- stage 4: DPO --------------------------------------------------------- #
    with timing.stage("dpo"):
        cfg = build_config({"dpo": args.dpo}, overrides + device_override[2:3])
        save_config(cfg, f"{cfg.dpo.out_dir}/config.yaml")
        dpo = run_dpo(cfg.dpo, full_config=cfg.to_dict())
        print(f"-> {dpo.last_ckpt}")

    # -- stage 5: eval -------------------------------------------------------- #
    with timing.stage("evaluate"):
        cfg = build_config({"eval": args.eval}, overrides + device_override[3:4])
        cfg.eval.checkpoint = str(sft.best_ckpt)  # judge the instruction-tuned model
        save_config(cfg, f"{cfg.eval.out_dir}/config.yaml")
        report = run_eval(cfg.eval, full_config=cfg.to_dict())
        print(
            f"-> val ppl {report['perplexity']['perplexity']:.1f} | judge({report['judge']['name']}) "
            f"grammar={report['judge']['grammar']:.1f} creativity={report['judge']['creativity']:.1f} "
            f"consistency={report['judge']['consistency']:.1f}"
        )

    # -- stage 6: generate ----------------------------------------------------- #
    with timing.stage("generate"):
        device_info = resolve_device(args.device)
        model, _ = load_model_from_checkpoint(dpo.last_ckpt, device_info.device)
        tokenizer = BPETokenizer.load(Path(cfg.eval.data_dir))
        sample = generate_text(
            model, tokenizer, "Once upon a time there was a little girl named Mia. One day she",
            max_new_tokens=100, temperature=0.8, top_k=40, device_info=device_info,
        )[0]

    _finish(timing, t0, gates_ok, sample)


def _finish(timing: Timing, t0: float, gates_ok: bool | None, sample) -> None:
    total = time.time() - t0
    print(f"\n{'=' * 72}\nPer-stage timing\n{'=' * 72}")
    print(format_timing_table(timing.rows))
    print(f"\nwall clock: {total / 60:.1f} min")

    if sample is not None:
        print(f"\n{'=' * 72}\nGenerated story continuation "
              f"({sample.n_new_tokens} tokens, {sample.tokens_per_sec:.0f} tok/s)\n{'=' * 72}")
        print(sample.text)
        print(
            "\n(Near-gibberish is the expected result at 1.5M params / 200 steps. "
            "The gates are the success criterion.)"
        )

    if gates_ok is None:
        print("\nPIPELINE OK: every stage ran. Gates were skipped (--skip-gates), so this is "
              "not a passing smoke test -- run `make smoke`.")
        return
    if not gates_ok:
        raise SystemExit("\nSMOKE TEST FAILED: one or more correctness gates did not pass")
    print(
        "\nGATES PASSED: all nine green." if sample is None
        else "\nSMOKE TEST PASSED: all nine gates green, every stage ran."
    )


if __name__ == "__main__":
    main()
