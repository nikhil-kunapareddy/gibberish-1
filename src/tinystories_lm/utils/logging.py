"""Stdout + JSONL logging, with W&B as an optional off-by-default sink.

Every training stage writes `<out_dir>/metrics.jsonl`: one JSON object per
logged event, machine-readable after the fact, no service required.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO


class RunLogger:
    def __init__(
        self,
        out_dir: str | Path,
        run_name: str,
        use_wandb: bool = False,
        wandb_project: str = "tinystories-lm",
        config: dict | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.path = self.out_dir / "metrics.jsonl"
        self._fh = self.path.open("a")
        self._stream = stream if stream is not None else sys.stdout
        self.t0 = time.time()
        self._wandb = None
        if use_wandb:
            try:
                import wandb  # noqa: PLC0415 - optional dependency, imported on demand
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "wandb=true but wandb is not installed; `pip install 'tinystories-lm[wandb]'`"
                ) from exc
            self._wandb = wandb.init(project=wandb_project, name=run_name, config=config or {})

    # -- events ------------------------------------------------------------- #

    def log(self, event: str, step: int | None = None, **fields: Any) -> None:
        """Record one event to JSONL (+ W&B) without printing it."""
        record = {
            "event": event,
            "run": self.run_name,
            "wall": round(time.time() - self.t0, 3),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if step is not None:
            record["step"] = step
        record.update(_jsonable(fields))
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if self._wandb is not None and event in ("train", "val"):
            self._wandb.log({f"{event}/{k}": v for k, v in fields.items()}, step=step)

    def metrics(self, event: str, step: int | None = None, **fields: Any) -> None:
        """Log an event and print a one-line summary."""
        self.log(event, step=step, **fields)
        head = f"[{self.run_name}] {event}"
        if step is not None:
            head += f" step {step}"
        body = " | ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        self.info(f"{head:<28} {body}")

    def info(self, message: str) -> None:
        print(message, file=self._stream, flush=True)

    def close(self) -> None:
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):  # torch scalar / numpy scalar
            try:
                v = v.item()
            except (ValueError, RuntimeError):
                v = str(v)
        out[k] = v
    return out


def format_timing_table(rows: list[tuple[str, float]], total_label: str = "TOTAL") -> str:
    """Render the per-stage timing table the smoke test ends with."""
    width = max([len(name) for name, _ in rows] + [len(total_label), 12])
    total = sum(sec for _, sec in rows)
    lines = [f"{'stage'.ljust(width)}  {'seconds':>9}  {'share':>6}", "-" * (width + 20)]
    for name, sec in rows:
        share = f"{100 * sec / total:5.1f}%" if total > 0 else "   n/a"
        lines.append(f"{name.ljust(width)}  {sec:9.2f}  {share:>6}")
    lines.append("-" * (width + 20))
    lines.append(f"{total_label.ljust(width)}  {total:9.2f}  {'100.0%':>6}")
    return "\n".join(lines)
