"""Validation perplexity.

Walks the validation stream in non-overlapping windows rather than sampling it,
so the number is a property of the corpus and not of an evaluation seed, and is
comparable across checkpoints and across runs.

Perplexity is reported as exp(token-weighted mean NLL). It is only comparable
between models that share a tokenizer -- a 8192-vocab BPE and GPT-2's 50257
vocab produce different token counts for the same text, so their perplexities
cannot be compared. The tokenizer hash in each checkpoint is what lets you
check that.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from ..data.loader import BatchSampler
from ..utils.device import autocast_ctx


@torch.no_grad()
def compute_perplexity(
    model,
    sampler: BatchSampler,
    max_batches: int | None = None,
    device_info=None,
    amp: bool = False,
) -> dict:
    """Token-weighted mean loss and perplexity over the split."""
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_batches = 0
    for x, y in sampler.sequential_batches(max_batches):
        with autocast_ctx(device_info, amp) if device_info else nullcontext():
            _, loss = model(x, y, reduction="mean")
        n_tokens = y.numel()
        total_nll += float(loss) * n_tokens
        total_tokens += n_tokens
        n_batches += 1
    model.train(was_training)

    if total_tokens == 0:
        raise ValueError(
            "no evaluation batches were produced: the split is shorter than "
            "block_size * batch_size"
        )
    mean_loss = total_nll / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "tokens": total_tokens,
        "batches": n_batches,
        "split": sampler.split,
    }
