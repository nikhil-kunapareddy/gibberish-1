"""RMSNorm and the SwiGLU MLP.

Both deviate from the TinyStories paper's GPT-Neo blocks (LayerNorm + GELU) and
both are chosen because they are what a modern scale-up expects; see README.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square norm: LayerNorm without the mean subtraction or bias.

    Normalisation is done in fp32 even under bf16 autocast -- the reciprocal
    square root of a bf16 sum of squares loses enough precision to matter.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"dim={self.weight.numel()}, eps={self.eps}"


class SwiGLU(nn.Module):
    """FFN(x) = W_down( silu(W_gate x) * W_up x ), no biases.

    Three matrices instead of two, so the hidden width is set to 8/3*d_model
    (see ModelConfig.hidden_dim) to keep the parameter count comparable to a
    4x-width GELU MLP.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.hidden_dim
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
