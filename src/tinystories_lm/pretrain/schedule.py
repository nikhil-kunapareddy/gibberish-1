"""Linear warmup then cosine decay.

The schedule is a pure function of the step, so "scheduler state" is just the
config plus the step count -- there is no hidden optimizer-coupled state to get
out of sync on resume. `state_dict()` exists so a checkpoint records the shape
of the curve that produced it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class CosineSchedule:
    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0 or self.warmup_steps > self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be in [0, total_steps={self.total_steps}]"
            )

    @property
    def min_lr(self) -> float:
        return self.base_lr * self.min_lr_ratio

    def lr_at(self, step: int) -> float:
        """LR for a 0-indexed step.

        Warmup is linear over the first `warmup_steps` and deliberately starts
        above zero (step 0 gets base_lr/warmup_steps rather than 0), so the
        first step is not a no-op.
        """
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / decay_steps
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.base_lr - self.min_lr)

    def apply(self, optimizer, step: int) -> float:
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict) -> "CosineSchedule":
        return cls(**state)
