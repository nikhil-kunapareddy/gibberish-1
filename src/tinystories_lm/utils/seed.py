"""Seeding and RNG-state capture.

Every stochastic source used by the pipeline (python `random`, numpy, torch CPU,
torch accelerator) is seeded from one integer and can be snapshotted into a
checkpoint so that `resume` is bit-identical to an uninterrupted run.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch from a single integer."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Off by default: it is slower, and some MPS/CUDA kernels have no
        # deterministic implementation. The gates that need exactness (resume
        # equivalence) run on CPU where the default kernels are already
        # deterministic.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict[str, Any]:
    """Snapshot every RNG this pipeline draws from."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        # MPS exposes a single global generator state.
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot from `get_rng_state`.

    Missing keys are tolerated: a checkpoint written on a CUDA box must still be
    resumable on a laptop, it just cannot restore the accelerator RNG.
    """
    if "python" in state:
        random.setstate(_as_tuple(state["python"]))
    if "numpy" in state:
        np.random.set_state(_as_tuple(state["numpy"]))
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])
        except (RuntimeError, ValueError):
            pass  # different GPU count than the machine that wrote the checkpoint
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(_as_byte_tensor(state["mps"]))


def _as_tuple(obj: Any) -> tuple:
    """torch.save/load round-trips tuples faithfully, but be liberal on input."""
    if isinstance(obj, list):
        return tuple(_as_tuple(o) if isinstance(o, list) else o for o in obj)
    return obj


def _as_byte_tensor(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.cpu().to(torch.uint8)
    return torch.tensor(obj, dtype=torch.uint8)
