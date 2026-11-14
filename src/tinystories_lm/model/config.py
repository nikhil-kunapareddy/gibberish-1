"""Model architecture config: a validated, serializable dataclass.

This object is embedded verbatim in every checkpoint, which is what makes a
checkpoint self-describing -- you can rebuild the exact model without knowing
anything about the run that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    """Decoder-only transformer hyperparameters.

    Deviates from the TinyStories paper (GPT-Neo): pre-norm + RMSNorm, RoPE
    instead of learned absolute positions, SwiGLU instead of GELU MLP, no bias
    terms, tied input/output embeddings. See README "Deviations from the paper".
    """

    vocab_size: int = 8192
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 128
    block_size: int = 256  # max sequence length the RoPE cache is built for

    # SwiGLU hidden width. Left as None it is 8/3 * d_model rounded up to a
    # multiple of `mlp_multiple_of`, which keeps a SwiGLU block's parameter
    # count level with a 4x GELU block's.
    mlp_hidden: int | None = None
    mlp_ratio: float = 8 / 3
    mlp_multiple_of: int = 64

    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02
    # Residual projections are scaled by 1/sqrt(2*n_layer) at init (GPT-2 trick)
    # so deep stacks do not start with an exploding residual stream.
    scale_residual_init: bool = True

    name: str = "unnamed"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.n_layer <= 0 or self.n_head <= 0 or self.d_model <= 0:
            raise ValueError("n_layer, n_head and d_model must all be positive")
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even: RoPE rotates dimension pairs"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.mlp_hidden is not None and self.mlp_hidden <= 0:
            raise ValueError(f"mlp_hidden must be positive, got {self.mlp_hidden}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def hidden_dim(self) -> int:
        """Resolved SwiGLU hidden width."""
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        raw = int(self.mlp_ratio * self.d_model)
        m = self.mlp_multiple_of
        return ((raw + m - 1) // m) * m

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig fields: {sorted(unknown)}")
        return cls(**d)
