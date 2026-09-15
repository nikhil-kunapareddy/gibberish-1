"""The decoder-only transformer: pre-norm blocks, tied embeddings, param counting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, KVCache, build_rope_cache
from .config import ModelConfig
from .layers import RMSNorm, SwiGLU

IGNORE_INDEX = -100  # target value that contributes nothing to the loss


class Block(nn.Module):
    """Pre-norm transformer block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Weight tying: the output projection *is* the embedding matrix.
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        # non-persistent: derived from config, so it stays out of checkpoints and
        # is rebuilt at the loading model's block_size.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            # Scale down whatever writes into the residual stream, so the
            # residual variance does not grow linearly with depth at init.
            if cfg.scale_residual_init and name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    # -- init ---------------------------------------------------------------- #

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # -- forward ------------------------------------------------------------- #

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        last_token_only: bool = False,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) int64 token ids. Returns (logits, loss).

        With a `cache`, `idx` holds only the *new* tokens and positions continue
        from `cache.pos`. `last_token_only` computes logits for the final
        position only -- the generation fast path, since the lm_head over an
        8k vocab is a large fraction of decode cost.
        """
        b, t = idx.shape
        start = cache.pos if cache is not None else 0
        if start + t > self.cfg.block_size:
            raise ValueError(
                f"sequence of length {start + t} exceeds block_size {self.cfg.block_size}"
            )

        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[start : start + t]
        sin = self.rope_sin[start : start + t]

        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.advance(t)

        x = self.final_norm(x)
        if last_token_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if last_token_only:
                raise ValueError("last_token_only=True cannot be combined with targets")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction=reduction,
            )
        return logits, loss

    # -- caches -------------------------------------------------------------- #

    def new_kv_cache(self, batch_size: int, max_seq_len: int | None = None) -> KVCache:
        max_seq_len = min(max_seq_len or self.cfg.block_size, self.cfg.block_size)
        param = next(self.parameters())
        return KVCache(
            n_layer=self.cfg.n_layer,
            batch_size=batch_size,
            n_head=self.cfg.n_head,
            head_dim=self.cfg.head_dim,
            max_seq_len=max_seq_len,
            device=param.device,
            dtype=param.dtype,
        )

    # -- introspection -------------------------------------------------------- #

    def num_params(self, include_embeddings: bool = True) -> int:
        """Parameter count. `include_embeddings=False` is the number people quote.

        With tied embeddings the token embedding is counted once; excluding it
        also excludes the (shared) output projection.
        """
        seen: set[int] = set()
        total = 0
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if not include_embeddings and self._is_embedding(name):
                continue
            total += p.numel()
        return total

    @staticmethod
    def _is_embedding(name: str) -> bool:
        return name.startswith("tok_emb.") or name.startswith("lm_head.")

    def param_breakdown(self) -> dict[str, int]:
        groups = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "other": 0}
        seen: set[int] = set()
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if self._is_embedding(name):
                key = "embedding"
            elif ".attn." in name:
                key = "attention"
            elif ".mlp." in name:
                key = "mlp"
            elif "norm" in name:
                key = "norm"
            else:
                key = "other"
            groups[key] += p.numel()
        groups["total"] = sum(v for k, v in groups.items() if k != "total")
        groups["non_embedding"] = groups["total"] - groups["embedding"]
        return groups

    def describe(self) -> str:
        cfg = self.cfg
        b = self.param_breakdown()
        lines = [
            f"Transformer[{cfg.name}]: "
            f"n_layer={cfg.n_layer} n_head={cfg.n_head} d_model={cfg.d_model} "
            f"hidden={cfg.hidden_dim} vocab={cfg.vocab_size} block={cfg.block_size} "
            f"tied={cfg.tie_embeddings}",
            f"  params total         {b['total']:>12,}  ({b['total'] / 1e6:.2f}M)",
            f"  params non-embedding {b['non_embedding']:>12,}  ({b['non_embedding'] / 1e6:.2f}M)",
            f"    embedding          {b['embedding']:>12,}",
            f"    attention          {b['attention']:>12,}",
            f"    mlp                {b['mlp']:>12,}",
            f"    norm               {b['norm']:>12,}",
        ]
        return "\n".join(lines)
