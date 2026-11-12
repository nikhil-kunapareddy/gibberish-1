"""Memory-mapped token streams and batch sampling.

The `.bin` files are flat uint16 token streams, memory-mapped so that a 500M
token corpus costs no RAM at load. There are no pre-built windows and no
padding: the sampler draws a random offset and takes `block_size + 1` tokens,
so changing context length is a config change and never a re-preparation of the
data.

The `+1` is the whole point: `x = tokens[i : i+T]` and `y = tokens[i+1 : i+T+1]`
come from the same slice, which is what gate 2 (off-by-one) checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .prepare import DTYPE, load_meta


class TokenStream:
    """Read-only memmap over one split's `.bin`."""

    def __init__(self, path: str | Path, expected_tokens: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"token stream not found: {self.path}")
        self.tokens = np.memmap(self.path, dtype=DTYPE, mode="r")
        if expected_tokens is not None and len(self.tokens) != expected_tokens:
            raise ValueError(
                f"{self.path} holds {len(self.tokens)} tokens, meta.json says {expected_tokens}"
            )

    def __len__(self) -> int:
        return int(len(self.tokens))

    def slice(self, start: int, length: int) -> np.ndarray:
        return np.asarray(self.tokens[start : start + length], dtype=np.int64)


class BatchSampler:
    """Uniform-random windows over a token stream.

    Randomness lives in one dedicated `torch.Generator`, whose state is saved
    in the checkpoint. That is what makes gate 7 (resume equivalence) exact:
    a resumed run draws the same batches an uninterrupted run would have.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        block_size: int,
        batch_size: int,
        device: torch.device | str = "cpu",
        seed: int = 1337,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        meta = load_meta(data_dir)
        if split not in meta["splits"]:
            raise ValueError(f"unknown split {split!r}; have {sorted(meta['splits'])}")
        info = meta["splits"][split]
        self.data_dir = Path(data_dir)
        self.split = split
        self.stream = TokenStream(self.data_dir / info["bin"], info["n_tokens"])
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.vocab_size = meta["tokenizer"]["vocab_size"]

        if len(self.stream) < block_size + 2:
            raise ValueError(
                f"{split}.bin has {len(self.stream)} tokens, too few for "
                f"block_size={block_size}. Prepare more stories or shrink block_size."
            )

        # TODO(scale): with world_size > 1 each rank reads a contiguous shard so
        # ranks never draw the same window in a step. Correct and a no-op at
        # world_size=1; the distributed sampler/all-reduce path is not built yet.
        self.rank, self.world_size = rank, world_size
        shard = len(self.stream) // world_size
        self.lo = rank * shard
        self.hi = len(self.stream) if rank == world_size - 1 else (rank + 1) * shard

        self.generator = torch.Generator()  # CPU generator: portable checkpoint state
        self.generator.manual_seed(seed + 1000 * rank)

    # -- sampling ------------------------------------------------------------- #

    @property
    def max_start(self) -> int:
        """Largest legal window start: needs block_size + 1 tokens after it."""
        return self.hi - self.block_size - 1

    def batch(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        bs = batch_size or self.batch_size
        starts = torch.randint(
            self.lo, self.max_start, (bs,), generator=self.generator, dtype=torch.int64
        )
        return self._gather(starts)

    def _gather(self, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.block_size
        windows = np.stack([self.stream.slice(int(s), t + 1) for s in starts])
        buf = torch.from_numpy(windows)
        x, y = buf[:, :-1].contiguous(), buf[:, 1:].contiguous()
        return self._to_device(x), self._to_device(y)

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def eval_batches(self, n_batches: int, seed: int = 0):
        """Deterministic batches for validation, independent of training RNG.

        Using the same seed every time means val loss at step 100 and step 200
        are measured on the same windows, so the curve is comparable.
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        for _ in range(n_batches):
            starts = torch.randint(
                self.lo, self.max_start, (self.batch_size,), generator=gen, dtype=torch.int64
            )
            yield self._gather(starts)

    def sequential_batches(self, n_batches: int | None = None):
        """Non-overlapping windows from the start of the split.

        Perplexity should be a property of the corpus, not of a sampling seed,
        so `eval/perplexity.py` walks the stream instead of sampling it.
        """
        t = self.block_size
        stride = t * self.batch_size
        total = (self.hi - self.lo - 1) // stride
        if n_batches is not None:
            total = min(total, n_batches)
        for b in range(total):
            base = self.lo + b * stride
            starts = torch.arange(base, base + stride, t, dtype=torch.int64)
            yield self._gather(starts)

    # -- checkpointing --------------------------------------------------------- #

    def state_dict(self) -> dict:
        return {
            "generator": self.generator.get_state(),
            "split": self.split,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("block_size") != self.block_size:
            raise ValueError(
                f"cannot resume: checkpoint sampled block_size={state.get('block_size')}, "
                f"this run uses {self.block_size}"
            )
        self.generator.set_state(_as_byte_tensor(state["generator"]))


def _as_byte_tensor(obj) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.cpu().to(torch.uint8)
    return torch.tensor(obj, dtype=torch.uint8)
