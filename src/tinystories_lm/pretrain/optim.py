"""AdamW with the usual weight-decay carve-outs.

Decay applies to matrices that do a linear map (attention and MLP projections)
and to nothing else: norms, biases, and embeddings are excluded. Embeddings in
particular are looked up sparsely, so decaying them shrinks the rows of rare
tokens toward zero for no reason -- and with tied embeddings that same matrix is
the output head.
"""

from __future__ import annotations

import inspect

import torch


def split_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        is_embedding = name.startswith(("tok_emb.", "lm_head.")) or "emb" in name
        if param.ndim >= 2 and not is_embedding:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay, "group": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group": "no_decay"},
    ]


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    groups = split_param_groups(model, weight_decay)
    kwargs = {}
    # Fused AdamW is a large win on CUDA and unavailable elsewhere.
    if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, **kwargs)


def describe_param_groups(model: torch.nn.Module, weight_decay: float) -> str:
    groups = split_param_groups(model, weight_decay)
    lines = []
    for g in groups:
        n_tensors = len(g["params"])
        n_params = sum(p.numel() for p in g["params"])
        lines.append(f"  {g['group']:<9} {n_tensors:>3} tensors, {n_params:>12,} params, wd={g['weight_decay']}")
    return "\n".join(lines)
