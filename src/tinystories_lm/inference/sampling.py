"""Token sampling: greedy, temperature, top-k, top-p.

Pure functions over a logits tensor, so each filter can be tested on its own
rather than only through a generation loop. Filters are applied in the order
temperature -> top-k -> top-p, which is the convention every other
implementation uses; changing the order changes what top_p means.
"""

from __future__ import annotations

import torch

NEG_INF = float("-inf")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0:
        return logits  # caller switches to greedy; scaling by 0 is undefined
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest-probability tokens, mask the rest. k <= 0 disables.

    Masks by index rather than by value threshold: with tied logits a
    `logits < threshold` filter keeps every tied token, so `top_k=2` can
    quietly sample from far more than 2 candidates.
    """
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    keep_idx = torch.topk(logits, k, dim=-1).indices
    keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, keep_idx, True)
    return logits.masked_fill(~keep, NEG_INF)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens whose mass exceeds p.

    The most probable token is always kept, even when it alone exceeds p.
    """
    if not 0.0 < p < 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # Drop everything after the token that first pushes the cumulative mass
    # past p; shifting right keeps that token itself.
    drop = cumulative - torch.softmax(sorted_logits, dim=-1) >= p
    drop[..., 0] = False
    return logits.masked_fill(drop.scatter(-1, sorted_idx, drop), NEG_INF)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: (B, V) -> next token ids: (B, 1)."""
    if logits.dim() != 2:
        raise ValueError(f"expected (B, V) logits, got shape {tuple(logits.shape)}")
    logits = logits.float()
    if greedy or temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = top_p_filter(top_k_filter(apply_temperature(logits, temperature), top_k), top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
