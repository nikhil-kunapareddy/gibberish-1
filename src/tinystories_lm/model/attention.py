"""Causal multi-head self-attention with RoPE and an optional KV cache.

The cache is the only thing standing between "generation is O(n^2) per token"
and "generation is O(n)". `tests/test_inference.py` asserts that cached and
uncached greedy decoding produce identical tokens, which is the only way to be
sure the cache is not quietly dropping or misaligning positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class KVCache:
    """Pre-allocated per-layer key/value buffers for incremental decoding."""

    def __init__(
        self,
        n_layer: int,
        batch_size: int,
        n_head: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (batch_size, n_head, max_seq_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.max_seq_len = max_seq_len
        self.pos = 0  # number of tokens already cached

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's keys/values for `layer`, return the full history.

        `pos` is advanced by the caller (`Transformer.forward`) once per forward
        pass, not per layer -- every layer sees the same window.
        """
        t = k.shape[2]
        if self.pos + t > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: {self.pos} cached + {t} new > {self.max_seq_len}. "
                f"Allocate the cache with a larger max_seq_len."
            )
        self.k[layer][:, :, self.pos : self.pos + t] = k.to(self.k[layer].dtype)
        self.v[layer][:, :, self.pos : self.pos + t] = v.to(self.v[layer].dtype)
        end = self.pos + t
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, t: int) -> None:
        self.pos += t

    def reset(self) -> None:
        self.pos = 0


def build_rope_cache(
    head_dim: int, max_seq_len: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables of shape (max_seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)  # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (x0, x1) pairs of the head dimension by the position angle.

    x:   (B, n_head, T, head_dim)
    cos: (T, head_dim/2), already sliced to this window's absolute positions.
    """
    b, h, t, d = x.shape
    x = x.float().view(b, h, t, d // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    cos = cos.view(1, 1, t, d // 2)
    sin = sin.view(1, 1, t, d // 2)
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.view(b, h, t, d)


def causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor | None:
    """Boolean attend-mask for a q_len window at the end of a k_len history.

    Returns None when no mask is needed (single-token decode attends to
    everything cached). `True` means "may attend".
    """
    if q_len == 1:
        return None
    offset = k_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1) + offset
    k_idx = torch.arange(k_len, device=device).unsqueeze(0)
    return k_idx <= q_idx


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv_proj(x).split(x.shape[-1], dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE uses absolute positions, which the cache tracks for us.
        q = apply_rope(q, cos, sin).to(v.dtype)
        k = apply_rope(k, cos, sin).to(v.dtype)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        k_len = k.shape[2]
        if t == k_len:
            # Full-sequence forward: let SDPA build the triangular mask itself
            # (it has a fused fast path for this).
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
            )
        else:
            mask = causal_mask(t, k_len, x.device)
            attn_mask = mask.view(1, 1, t, k_len) if mask is not None else None
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
            )

        y = y.transpose(1, 2).contiguous().view(b, t, self.n_head * self.head_dim)
        return self.resid_dropout(self.out_proj(y))
