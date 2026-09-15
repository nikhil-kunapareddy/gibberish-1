"""Self-describing checkpoints.

A checkpoint must be loadable with no external knowledge of how it was trained,
so every one carries:

  * model weights **and** the full `ModelConfig` that shaped them
  * optimizer state and scheduler state (step-parameterised, see pretrain/schedule.py)
  * the step count, best-val-loss so far, and the stage that wrote it
  * RNG state for python/numpy/torch/accelerator **and** the data sampler's own
    generator, which is what makes `resume` bit-identical to a straight run
  * the tokenizer hash + the data meta hash, so a mismatched tokenizer is caught
    at load time instead of producing quiet gibberish

`load_model_from_checkpoint(path)` is therefore all you need to get a working
model back from a bare .pt file.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from ..model.config import ModelConfig
from ..model.transformer import Transformer
from .seed import get_rng_state, set_rng_state

FORMAT_VERSION = 1
_STEP_RE = re.compile(r"ckpt_step_(\d+)\.pt$")


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """sha256 of a file, truncated to 16 hex chars (enough to catch mismatches)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_config: ModelConfig,
    step: int,
    stage: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler_state: dict | None = None,
    data_state: dict | None = None,
    tokenizer: dict | None = None,
    data_meta: dict | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
    best_val_loss: float | None = None,
) -> Path:
    """Write one checkpoint atomically (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "step": step,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model_config.to_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler_state,
        "data_state": data_state,
        "rng_state": get_rng_state(),
        "tokenizer": tokenizer or {},
        "data_meta": data_meta or {},
        "config": config or {},
        "metrics": metrics or {},
        "best_val_loss": best_val_loss,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n"
            f"  Did an earlier stage fail? Each stage reads the previous stage's "
            f"checkpoint from disk."
        )
    # weights_only=False: our payload holds RNG state tuples and config dicts,
    # not just tensors. Only load checkpoints you produced.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state" not in ckpt or "model_config" not in ckpt:
        raise ValueError(f"{path} is not a tinystories-lm checkpoint (no model_state/model_config)")
    version = ckpt.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} has checkpoint format_version={version}, this code writes {FORMAT_VERSION}"
        )
    return ckpt


def load_model_from_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    eval_mode: bool = True,
) -> tuple[Transformer, dict]:
    """Rebuild the exact model a checkpoint describes. No other inputs needed."""
    ckpt = load_checkpoint(path, map_location="cpu")
    model_config = ModelConfig.from_dict(ckpt["model_config"])
    model = Transformer(model_config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    if eval_mode:
        model.eval()
    return model, ckpt


def restore_training_state(
    ckpt: dict,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> int:
    """Load weights/optimizer/RNG back into live objects. Returns the step count."""
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if restore_rng and ckpt.get("rng_state"):
        set_rng_state(ckpt["rng_state"])
    return int(ckpt["step"])


def assert_tokenizer_match(ckpt: dict, tokenizer_hash: str, where: str = "") -> None:
    """Fail loudly when a checkpoint meets a tokenizer it was not trained with."""
    recorded = (ckpt.get("tokenizer") or {}).get("hash")
    if recorded and tokenizer_hash and recorded != tokenizer_hash:
        raise ValueError(
            f"tokenizer mismatch{' in ' + where if where else ''}: checkpoint was trained with "
            f"tokenizer hash {recorded}, but the data dir provides {tokenizer_hash}. "
            f"Re-run prepare_data or point at the matching checkpoint."
        )


class CheckpointManager:
    """Writes `ckpt_step_*.pt`, keeps the last K, and mirrors last/best/early."""

    def __init__(self, out_dir: str | Path, keep_last_k: int = 2, stage: str = "pretrain") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = max(1, keep_last_k)
        self.stage = stage
        self.best_val_loss = float("inf")

    # -- paths -------------------------------------------------------------- #

    def step_path(self, step: int) -> Path:
        return self.out_dir / f"ckpt_step_{step:06d}.pt"

    @property
    def last_path(self) -> Path:
        return self.out_dir / "ckpt_last.pt"

    @property
    def best_path(self) -> Path:
        return self.out_dir / "ckpt_best.pt"

    @property
    def early_path(self) -> Path:
        return self.out_dir / "ckpt_early.pt"

    # -- writing ------------------------------------------------------------ #

    def save(self, step: int, *, val_loss: float | None = None, alias: str | None = None, **kw) -> Path:
        """Save a step checkpoint, update last/best aliases, rotate old files."""
        path = save_checkpoint(
            self.step_path(step),
            step=step,
            stage=self.stage,
            best_val_loss=self.best_val_loss if self.best_val_loss < float("inf") else None,
            **kw,
        )
        shutil.copyfile(path, self.last_path)
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            shutil.copyfile(path, self.best_path)
        if alias:
            shutil.copyfile(path, self.out_dir / f"ckpt_{alias}.pt")
        self._rotate()
        return path

    def _rotate(self) -> None:
        steps = sorted(
            (int(m.group(1)), p)
            for p in self.out_dir.glob("ckpt_step_*.pt")
            if (m := _STEP_RE.search(p.name))
        )
        for _, p in steps[: max(0, len(steps) - self.keep_last_k)]:
            p.unlink(missing_ok=True)

    # -- reading ------------------------------------------------------------ #

    def latest(self) -> Path | None:
        if self.last_path.exists():
            return self.last_path
        steps = sorted(
            (int(m.group(1)), p)
            for p in self.out_dir.glob("ckpt_step_*.pt")
            if (m := _STEP_RE.search(p.name))
        )
        return steps[-1][1] if steps else None

    def resolve_resume(self, resume: str) -> Path | None:
        """Interpret the `resume` config value: "" | "auto" | explicit path."""
        if not resume:
            return None
        if resume == "auto":
            return self.latest()
        p = Path(resume)
        if not p.exists():
            raise FileNotFoundError(f"resume={resume!r} does not exist")
        return p


def describe(path: str | Path) -> str:
    """Human-readable summary of a checkpoint -- used by `scripts/evaluate.py -v`."""
    ckpt = load_checkpoint(path)
    mc = ckpt["model_config"]
    return json.dumps(
        {
            "stage": ckpt["stage"],
            "step": ckpt["step"],
            "created_at": ckpt["created_at"],
            "model": {k: mc[k] for k in ("n_layer", "n_head", "d_model", "vocab_size", "block_size")},
            "tokenizer": ckpt.get("tokenizer", {}),
            "metrics": ckpt.get("metrics", {}),
            "best_val_loss": ckpt.get("best_val_loss"),
        },
        indent=2,
    )
