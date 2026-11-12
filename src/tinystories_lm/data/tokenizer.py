"""Train and load the byte-level BPE tokenizer.

Why not GPT-2's 50257-entry vocab: at d_model=512 a 50k vocab is ~25.7M
embedding parameters, which would be most of a "33M" model. At 8192 the
embedding is 4.2M and the transformer stack is the model. `vocab_size` is a
config value and the tokenizer is swappable -- everything downstream reads the
vocab size out of `meta.json` / the checkpoint, never a constant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

EOT = "<|endoftext|>"
TOKENIZER_FILENAME = "tokenizer.json"


class BPETokenizer:
    """Thin wrapper over `tokenizers.Tokenizer` with the pipeline's conventions."""

    def __init__(self, tokenizer: Tokenizer, path: str | Path | None = None) -> None:
        self.tok = tokenizer
        self.path = Path(path) if path else None
        eot = self.tok.token_to_id(EOT)
        if eot is None:
            raise ValueError(f"tokenizer is missing the {EOT} special token")
        self.eot_id = eot

    # -- io ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        if path.is_dir():
            path = path / TOKENIZER_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"tokenizer not found: {path}. Run scripts/prepare_data.py first."
            )
        return cls(Tokenizer.from_file(str(path)), path)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / TOKENIZER_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))
        self.path = path
        return path

    # -- properties ----------------------------------------------------------- #

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def hash(self) -> str:
        """Stable hash of the tokenizer's serialized form.

        Checkpoints record it; loaders compare it. A model decoded with a
        different tokenizer produces plausible-looking nonsense, so this is
        worth catching mechanically.
        """
        from ..utils.checkpoint import text_hash

        return text_hash(self.tok.to_str())

    def meta(self) -> dict:
        return {
            "hash": self.hash(),
            "vocab_size": self.vocab_size,
            "eot_id": self.eot_id,
            "eot_token": EOT,
            "path": str(self.path) if self.path else None,
        }

    # -- encode / decode ------------------------------------------------------ #

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        ids = self.tok.encode(text).ids
        if add_eot:
            ids.append(self.eot_id)
        return ids

    def encode_batch(self, texts: Sequence[str], add_eot: bool = False) -> list[list[int]]:
        out = [e.ids for e in self.tok.encode_batch(list(texts))]
        if add_eot:
            for ids in out:
                ids.append(self.eot_id)
        return out

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    out_path: str | Path,
    min_frequency: int = 2,
    show_progress: bool = True,
) -> BPETokenizer:
    """Train a byte-level BPE tokenizer.

    Byte-level is what makes the roundtrip gate (gate 1) achievable: every byte
    sequence is representable, so encode->decode is exact for any input,
    including characters never seen in training.
    """
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # add_prefix_space=False keeps encode/decode exactly inverse; use_regex=True
    # gives the GPT-2 style word-boundary splitting.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(_as_iterator(texts), trainer=trainer)

    bpe = BPETokenizer(tokenizer)
    bpe.save(out_path)
    return bpe


def _as_iterator(texts: Iterable[str]) -> Iterator[str]:
    return iter(texts)


def load_tokenizer_meta(data_dir: str | Path) -> dict:
    """Read the tokenizer block out of a prepared data directory's meta.json."""
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found: {data_dir} is not a prepared data directory. "
            f"Run scripts/prepare_data.py --data <config> first."
        )
    return json.loads(meta_path.read_text())["tokenizer"]
