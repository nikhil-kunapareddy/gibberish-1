"""Raw text -> flat uint16 token stream + meta.json.

Output contract (everything downstream reads exactly this):

    <out_dir>/train.bin        uint16 token ids, one <|endoftext|> after each story
    <out_dir>/val.bin          same
    <out_dir>/tokenizer.json   the BPE trained here, on the *train* split only
    <out_dir>/sft_train.jsonl  {"prompt", "completion", "hash"} from -Instruct
    <out_dir>/sft_val.jsonl
    <out_dir>/meta.json        token counts, tokenizer hash, and the sorted
                               story hashes of each split

Two decisions worth stating:

* **Split by story, not by token.** Tokens are concatenated *after* the split,
  so no story can straddle the boundary. Assignment is a hash of the story text
  (salted by `split_seed`), so it is reproducible, order-independent, and
  auditable after the fact -- the hashes go in meta.json.
* **The tokenizer sees train stories only.** Training BPE merges on validation
  text is a small but real leak.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm

from .download import ensure_corpora
from .tokenizer import EOT, BPETokenizer, train_tokenizer

META_VERSION = 1
DTYPE = np.uint16
ENCODE_CHUNK = 2000  # stories per encode_batch call


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_stories(text: str, drop_truncated_tail: bool = True) -> list[str]:
    """Split a TinyStories text dump into individual stories.

    The upstream *-valid.txt files begin mid-story (the first record has no
    beginning), and a range-limited download ends mid-story, so partial records
    at both ends are dropped rather than fed to the tokenizer as if whole.
    """
    records = [r.strip() for r in text.split(EOT)]
    if drop_truncated_tail and records:
        records = records[:-1]  # the tail after the last EOT is incomplete
    stories = [r for r in records if r]
    if stories and _starts_mid_sentence(stories[0]):
        stories = stories[1:]
    return stories


def _starts_mid_sentence(text: str) -> bool:
    head = text.lstrip()
    return bool(head) and not (head[0].isupper() or head[0] in "\"'“‘")


@dataclass(frozen=True)
class InstructExample:
    """One TinyStories-Instruct record split at the `Story:` marker.

    The prompt is the instruction block (Features / Words / Summary / Random
    sentence, in whatever order that record used) plus the `Story:` marker; the
    completion is the story itself. SFT masks the loss to the completion.
    """

    prompt: str
    completion: str
    hash: str


_INSTRUCT_FIELDS = ("Features:", "Words:", "Summary:", "Random sentence:")
_STORY_MARKER = "Story:"


def parse_instruct(text: str, drop_truncated_tail: bool = True) -> list[InstructExample]:
    records = [r.strip("\n") for r in text.split(EOT)]
    if drop_truncated_tail and records:
        records = records[:-1]
    out: list[InstructExample] = []
    for record in records:
        example = _parse_instruct_record(record)
        if example is not None:
            out.append(example)
    return out


def _parse_instruct_record(record: str) -> InstructExample | None:
    record = record.strip("\n")
    if not record.strip():
        return None
    idx = record.find(_STORY_MARKER)
    if idx < 0:
        return None
    prompt = record[: idx + len(_STORY_MARKER)]
    completion = record[idx + len(_STORY_MARKER) :].strip()
    # A record whose instruction block was cut off (first record of a valid
    # file, or a range-truncated tail) starts with no field label; skip it.
    if not prompt.lstrip().startswith(_INSTRUCT_FIELDS):
        return None
    if not completion.strip():
        return None
    return InstructExample(prompt=prompt + "\n", completion=completion, hash=story_hash(completion))


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def story_hash(text: str, salt: int | str = "") -> str:
    """Stable 16-hex-char content hash, used both as identity and as the split key."""
    h = hashlib.sha256(f"{salt}\x00{text.strip()}".encode("utf-8"))
    return h.hexdigest()[:16]


def assign_split(hash_hex: str, val_fraction: float, buckets: int = 10_000) -> str:
    """Deterministic per-story split from its hash. Duplicates land together."""
    bucket = int(hash_hex, 16) % buckets
    return "val" if bucket < int(val_fraction * buckets) else "train"


def split_stories(
    stories: Iterable[str], val_fraction: float, split_seed: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Returns ({split: stories}, {split: sorted hashes}), duplicates removed."""
    texts: dict[str, list[str]] = {"train": [], "val": []}
    hashes: dict[str, list[str]] = {"train": [], "val": []}
    seen: set[str] = set()
    for story in stories:
        h = story_hash(story, salt=split_seed)
        if h in seen:
            continue  # exact duplicate: keep one copy, and never across splits
        seen.add(h)
        split = assign_split(h, val_fraction)
        texts[split].append(story)
        hashes[split].append(h)
    return texts, {k: sorted(v) for k, v in hashes.items()}


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_token_bin(
    stories: list[str], tokenizer: BPETokenizer, path: str | Path, quiet: bool = False
) -> int:
    """Encode stories and append them to one flat uint16 file. Returns token count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer.vocab_size > np.iinfo(DTYPE).max + 1:
        raise ValueError(
            f"vocab_size {tokenizer.vocab_size} does not fit in {DTYPE.__name__}; "
            f"widen the dtype in data/prepare.py and data/loader.py together"
        )
    total = 0
    with path.open("wb") as f:
        for chunk in tqdm(
            list(_chunks(stories, ENCODE_CHUNK)),
            desc=f"  encoding {path.name}",
            disable=quiet,
            leave=False,
        ):
            encoded = tokenizer.encode_batch(chunk, add_eot=True)
            flat = np.fromiter(
                (tok for ids in encoded for tok in ids), dtype=DTYPE, count=sum(map(len, encoded))
            )
            flat.tofile(f)
            total += int(flat.size)
    return total


def _chunks(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_data.py first")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def prepare(cfg, quiet: bool = False) -> dict:
    """Run the whole data stage for a DataConfig. Returns the meta.json contents."""
    t0 = time.time()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if quiet else print

    log(f"[prepare] out_dir={out_dir}")
    files = ensure_corpora(cfg, quiet=quiet)

    # -- stories -> splits -------------------------------------------------- #
    raw = files["stories"].path.read_text(encoding="utf-8", errors="replace")
    stories = parse_stories(raw, drop_truncated_tail=True)
    log(f"[prepare] parsed {len(stories):,} stories from {files['stories'].path.name}")
    if cfg.num_stories:
        stories = stories[: cfg.num_stories]
    if not stories:
        raise RuntimeError(
            f"no stories parsed from {files['stories'].path}; "
            f"is max_download_bytes too small?"
        )
    texts, hashes = split_stories(stories, cfg.val_fraction, cfg.split_seed)
    log(f"[prepare] split: train={len(texts['train']):,} val={len(texts['val']):,} stories")
    if not texts["train"] or not texts["val"]:
        raise RuntimeError(
            f"a split came out empty (train={len(texts['train'])}, val={len(texts['val'])}); "
            f"increase num_stories or val_fraction"
        )

    # -- tokenizer (train split only) --------------------------------------- #
    corpus = texts["train"][: cfg.tokenizer_train_stories]
    log(f"[prepare] training {cfg.vocab_size}-token BPE on {len(corpus):,} train stories")
    tokenizer = train_tokenizer(
        corpus, vocab_size=cfg.vocab_size, out_path=out_dir, show_progress=not quiet
    )
    log(f"[prepare] tokenizer vocab={tokenizer.vocab_size} hash={tokenizer.hash()}")

    # -- token streams ------------------------------------------------------- #
    n_tokens = {
        split: write_token_bin(texts[split], tokenizer, out_dir / f"{split}.bin", quiet=quiet)
        for split in ("train", "val")
    }
    log(f"[prepare] tokens: train={n_tokens['train']:,} val={n_tokens['val']:,}")

    # -- instruction data for SFT ------------------------------------------- #
    instruct_raw = files["instruct"].path.read_text(encoding="utf-8", errors="replace")
    examples = parse_instruct(instruct_raw, drop_truncated_tail=True)
    if cfg.num_instruct_examples:
        examples = examples[: cfg.num_instruct_examples]
    sft_counts = {"train": 0, "val": 0}
    sft_rows: dict[str, list[dict]] = {"train": [], "val": []}
    for ex in examples:
        split = assign_split(story_hash(ex.completion, salt=cfg.split_seed), cfg.val_fraction)
        sft_rows[split].append(
            {"prompt": ex.prompt, "completion": ex.completion, "hash": ex.hash}
        )
    for split in ("train", "val"):
        sft_counts[split] = write_jsonl(sft_rows[split], out_dir / f"sft_{split}.jsonl")
    log(f"[prepare] instruct examples: train={sft_counts['train']:,} val={sft_counts['val']:,}")
    if sft_counts["train"] == 0:
        raise RuntimeError("no TinyStories-Instruct examples parsed; SFT would have no data")

    # -- meta ---------------------------------------------------------------- #
    meta = {
        "meta_version": META_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "config": {
            k: getattr(cfg, k)
            for k in (
                "name", "dataset", "instruct_dataset", "source_file", "instruct_source_file",
                "num_stories", "num_instruct_examples", "vocab_size", "val_fraction",
                "split_seed", "tokenizer_train_stories", "max_download_bytes",
            )
        },
        "dtype": DTYPE.__name__,
        "tokenizer": tokenizer.meta(),
        "splits": {
            split: {
                "bin": f"{split}.bin",
                "n_tokens": n_tokens[split],
                "n_stories": len(texts[split]),
                "story_hashes": hashes[split],  # sorted; gate 3 reads these
            }
            for split in ("train", "val")
        },
        "sft": {
            "train_file": "sft_train.jsonl",
            "val_file": "sft_val.jsonl",
            "n_train": sft_counts["train"],
            "n_val": sft_counts["val"],
        },
        "sources": {
            name: {
                "path": str(f.path),
                "bytes_read": f.bytes_read,
                "total_bytes": f.total_bytes,
                "truncated": f.truncated,
            }
            for name, f in files.items()
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[prepare] wrote {out_dir / 'meta.json'} in {meta['elapsed_sec']}s")
    return meta


# --------------------------------------------------------------------------- #
# Input contract for downstream stages
# --------------------------------------------------------------------------- #


def load_meta(data_dir: str | Path) -> dict:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found.\n"
            f"  {data_dir} is not a prepared data directory. Run:\n"
            f"    python scripts/prepare_data.py --data configs/data/smoke.yaml"
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("meta_version") != META_VERSION:
        raise ValueError(
            f"{meta_path} has meta_version={meta.get('meta_version')}, "
            f"this code expects {META_VERSION}; re-run prepare_data"
        )
    return meta


def validate_data_dir(data_dir: str | Path, need_sft: bool = False) -> dict:
    """Validate a stage's input contract, with an actionable message on failure."""
    data_dir = Path(data_dir)
    meta = load_meta(data_dir)
    required = ["train.bin", "val.bin", "tokenizer.json"]
    if need_sft:
        required += [meta["sft"]["train_file"], meta["sft"]["val_file"]]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"data dir {data_dir} is missing {missing}. Re-run scripts/prepare_data.py."
        )
    for split in ("train", "val"):
        expected = meta["splits"][split]["n_tokens"] * np.dtype(DTYPE).itemsize
        actual = (data_dir / f"{split}.bin").stat().st_size
        if actual != expected:
            raise ValueError(
                f"{data_dir / f'{split}.bin'} is {actual} bytes but meta.json says "
                f"{expected} ({meta['splits'][split]['n_tokens']} {DTYPE.__name__} tokens). "
                f"The data dir is inconsistent; re-run prepare_data."
            )
    return meta
