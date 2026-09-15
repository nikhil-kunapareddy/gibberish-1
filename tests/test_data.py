"""Gates 1-3: tokenizer roundtrip, off-by-one, split leakage."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tinystories_lm.data.loader import BatchSampler, TokenStream
from tinystories_lm.data.prepare import (
    assign_split,
    parse_instruct,
    parse_stories,
    split_stories,
    story_hash,
)
from tinystories_lm.data.tokenizer import EOT

# --------------------------------------------------------------------------- #
# GATE 1: tokenizer roundtrip
# --------------------------------------------------------------------------- #


def test_gate1_tokenizer_roundtrip_1000_stories(tokenizer, raw_stories):
    """encode -> decode must reproduce 1000 random stories byte for byte."""
    rng = random.Random(0)
    sample = rng.sample(raw_stories, min(1000, len(raw_stories)))
    assert len(sample) == 1000, f"only {len(sample)} stories available"

    failures = []
    for story in sample:
        decoded = tokenizer.decode(tokenizer.encode(story))
        if decoded != story:
            failures.append((story[:60], decoded[:60]))
    assert not failures, (
        f"{len(failures)}/{len(sample)} stories did not survive the roundtrip; "
        f"first: {failures[0]}"
    )


def test_gate1_roundtrip_handles_unseen_characters(tokenizer):
    """Byte-level BPE must be lossless even on text the corpus never contained."""
    for text in ["emoji: 🐉🔥", "日本語のテキスト", "tabs\tand\nnewlines", "  leading space"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_eot_is_a_single_token(tokenizer):
    ids = tokenizer.encode(EOT)
    assert ids == [tokenizer.eot_id], f"{EOT} encoded to {ids}, expected one special token"


def test_tokenizer_hash_is_stable(tokenizer, prepared_dir):
    from tinystories_lm.data.tokenizer import BPETokenizer

    assert BPETokenizer.load(prepared_dir).hash() == tokenizer.hash()


# --------------------------------------------------------------------------- #
# GATE 2: off-by-one
# --------------------------------------------------------------------------- #


def test_gate2_targets_are_inputs_shifted_by_one(prepared_dir):
    """y[i, t] must equal x[i, t+1] for every element of the batch."""
    sampler = BatchSampler(prepared_dir, "train", block_size=64, batch_size=8, seed=0)
    for _ in range(5):
        x, y = sampler.batch()
        assert x.shape == y.shape == (8, 64)
        assert torch.equal(y[:, :-1], x[:, 1:]), "targets are not inputs shifted left by one"


def test_gate2_holds_for_eval_and_sequential_batches(prepared_dir):
    sampler = BatchSampler(prepared_dir, "val", block_size=32, batch_size=4, seed=0)
    for x, y in sampler.eval_batches(3, seed=7):
        assert torch.equal(y[:, :-1], x[:, 1:])
    for x, y in sampler.sequential_batches(3):
        assert torch.equal(y[:, :-1], x[:, 1:])


def test_gate2_last_target_is_the_next_real_token(prepared_dir, meta):
    """The final target comes from the stream, not from wrap-around or padding."""
    stream = TokenStream(prepared_dir / "train.bin", meta["splits"]["train"]["n_tokens"])
    block = 16
    starts = torch.tensor([0, 5, 123, len(stream) - block - 1])
    sampler = BatchSampler(prepared_dir, "train", block_size=block, batch_size=4, seed=0)
    x, y = sampler._gather(starts)
    for row, start in enumerate(starts.tolist()):
        window = stream.slice(start, block + 1)
        assert x[row].tolist() == window[:-1].tolist()
        assert y[row].tolist() == window[1:].tolist()


def test_sampler_never_reads_past_the_end(prepared_dir):
    sampler = BatchSampler(prepared_dir, "val", block_size=64, batch_size=16, seed=3)
    assert sampler.max_start + sampler.block_size + 1 <= len(sampler.stream)
    for _ in range(20):
        sampler.batch()  # would raise/garbage on an out-of-range slice


def test_sampler_state_is_restorable(prepared_dir):
    a = BatchSampler(prepared_dir, "train", block_size=32, batch_size=4, seed=11)
    a.batch()
    state = a.state_dict()
    expected = a.batch()
    b = BatchSampler(prepared_dir, "train", block_size=32, batch_size=4, seed=999)
    b.load_state_dict(state)
    got = b.batch()
    assert torch.equal(expected[0], got[0]) and torch.equal(expected[1], got[1])


# --------------------------------------------------------------------------- #
# GATE 3: no split leakage
# --------------------------------------------------------------------------- #


def test_gate3_train_and_val_story_hashes_are_disjoint(meta):
    train = set(meta["splits"]["train"]["story_hashes"])
    val = set(meta["splits"]["val"]["story_hashes"])
    assert train and val, "a split is empty"
    assert train & val == set(), f"{len(train & val)} stories appear in both splits"


def test_gate3_hashes_are_sorted_and_unique(meta):
    for split in ("train", "val"):
        hashes = meta["splits"][split]["story_hashes"]
        assert hashes == sorted(hashes), f"{split} hashes are not sorted (not auditable)"
        assert len(hashes) == len(set(hashes)), f"{split} contains duplicate stories"
        assert len(hashes) == meta["splits"][split]["n_stories"]


def test_gate3_split_is_reproducible_and_order_independent(raw_stories):
    stories = raw_stories[:500]
    a, ha = split_stories(stories, 0.1, split_seed=42)
    b, hb = split_stories(list(reversed(stories)), 0.1, split_seed=42)
    assert ha == hb, "split assignment depends on input order"
    assert set(a["val"]) == set(b["val"])


def test_gate3_duplicate_stories_cannot_straddle_the_split(raw_stories):
    duplicated = raw_stories[:200] + raw_stories[:200]
    texts, hashes = split_stories(duplicated, 0.1, split_seed=7)
    assert set(hashes["train"]) & set(hashes["val"]) == set()
    assert len(hashes["train"]) + len(hashes["val"]) == 200, "duplicates were not collapsed"


def test_gate3_val_fraction_is_respected(raw_stories):
    _, hashes = split_stories(raw_stories[:4000], 0.1, split_seed=1234)
    n_val = len(hashes["val"])
    total = n_val + len(hashes["train"])
    assert 0.07 < n_val / total < 0.13, f"val fraction drifted: {n_val}/{total}"


def test_split_assignment_is_a_pure_function_of_the_hash():
    h = story_hash("hello", salt=1)
    assert assign_split(h, 0.5) == assign_split(h, 0.5)
    assert assign_split(h, 0.0) == "train", "val_fraction=0 must put everything in train"
    assert assign_split(h, 1.0) == "val", "val_fraction=1 must put everything in val"
    # Bucket 0 is the first val bucket; bucket 9999 the last train bucket.
    assert assign_split(f"{0:016x}", 0.5) == "val"
    assert assign_split(f"{9999:016x}", 0.5) == "train"


# --------------------------------------------------------------------------- #
# Parsing and the on-disk contract
# --------------------------------------------------------------------------- #


def test_parse_stories_drops_partial_records_at_both_ends():
    text = f"broken tail of a story\n{EOT}\nOnce there was a cat.\n{EOT}\nA truncated st"
    assert parse_stories(text) == ["Once there was a cat."]


def test_parse_instruct_splits_prompt_from_completion():
    record = (
        "Features: Dialogue\nSummary: A cat naps.\nStory: \nThe cat sat. It napped.\n"
        f"{EOT}\ntrailing"
    )
    examples = parse_instruct(record)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.prompt.rstrip().endswith("Story:")
    assert "Summary: A cat naps." in ex.prompt
    assert ex.completion == "The cat sat. It napped."
    assert "The cat sat" not in ex.prompt, "the story leaked into the prompt"


def test_parse_instruct_skips_records_with_no_instruction_block():
    assert parse_instruct(f"...cut off mid story\nStory: \nmore text\n{EOT}\nx") == []


def test_bins_are_uint16_and_match_meta(prepared_dir, meta):
    for split in ("train", "val"):
        raw = np.memmap(prepared_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        assert len(raw) == meta["splits"][split]["n_tokens"]
        assert int(raw.max()) < meta["tokenizer"]["vocab_size"]


def test_every_story_is_terminated_by_eot(prepared_dir, meta):
    """One <|endoftext|> per story, so story boundaries survive concatenation."""
    eot = meta["tokenizer"]["eot_id"]
    for split in ("train", "val"):
        raw = np.memmap(prepared_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        assert int((np.asarray(raw) == eot).sum()) == meta["splits"][split]["n_stories"]
        assert int(raw[-1]) == eot, "stream does not end at a story boundary"


def test_validate_data_dir_rejects_an_inconsistent_dir(prepared_dir, tmp_path):
    import shutil

    from tinystories_lm.data.prepare import validate_data_dir

    validate_data_dir(prepared_dir, need_sft=True)  # the real one passes

    broken = tmp_path / "broken"
    shutil.copytree(prepared_dir, broken)
    (broken / "val.bin").write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError, match="inconsistent"):
        validate_data_dir(broken)


# --------------------------------------------------------------------------- #
# Shipped configs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "section", ["model", "data", "pretrain", "sft", "dpo", "eval"]
)
def test_every_shipped_config_loads_and_validates(repo_root, section):
    """A broken YAML should fail here, not thirty minutes into a run."""
    from tinystories_lm.utils.config import build_config

    paths = sorted((repo_root / "configs" / section).glob("*.yaml"))
    assert paths, f"configs/{section}/ is empty"
    for path in paths:
        cfg = build_config({section: str(path)})
        assert getattr(cfg[section], "name", "unnamed") != "unnamed", f"{path} has no name"


def test_full_and_smoke_configs_are_consistent(repo_root):
    """The full configs must point at each other's outputs, not the smoke run's."""
    from tinystories_lm.utils.config import build_config

    full = build_config({
        "data": str(repo_root / "configs/data/full.yaml"),
        "pretrain": str(repo_root / "configs/pretrain/full.yaml"),
        "sft": str(repo_root / "configs/sft/full.yaml"),
        "dpo": str(repo_root / "configs/dpo/full.yaml"),
        "eval": str(repo_root / "configs/eval/full.yaml"),
    })
    assert full.pretrain.data_dir == full.data.out_dir
    assert full.sft.data_dir == full.data.out_dir
    assert full.sft.init_from.startswith(full.pretrain.out_dir)
    assert full.dpo.init_from.startswith(full.sft.out_dir)
    assert full.dpo.sampler_ckpt.startswith(full.pretrain.out_dir)
    assert full.eval.data_dir == full.data.out_dir
    assert full.eval.block_size <= 512  # must fit the model's block_size
