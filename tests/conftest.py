"""Shared fixtures.

Tests run against a small *real* prepared dataset rather than synthetic tokens:
the gates exist to catch pipeline bugs, and a fake corpus would route around
exactly the code being tested. The raw download is cached in `data/raw`, so
only the first run touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinystories_lm.data.prepare import parse_stories, prepare  # noqa: E402
from tinystories_lm.utils.config import build_config  # noqa: E402

TEST_STORIES = 600
TEST_VOCAB = 1024


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def prepared_dir(tmp_path_factory) -> Path:
    """A small prepared data dir: 600 stories, 1024-token vocab."""
    out = tmp_path_factory.mktemp("prepared")
    cfg = build_config(
        {"data": str(REPO_ROOT / "configs/data/smoke.yaml")},
        [
            f"--data.out_dir={out}",
            f"--data.data_dir={REPO_ROOT / 'data'}",
            f"--data.num_stories={TEST_STORIES}",
            f"--data.vocab_size={TEST_VOCAB}",
            "--data.num_instruct_examples=200",
            f"--data.tokenizer_train_stories={TEST_STORIES}",
            "--data.val_fraction=0.1",
        ],
    )
    prepare(cfg.data, quiet=True)
    return out


@pytest.fixture(scope="session")
def meta(prepared_dir) -> dict:
    import json

    return json.loads((prepared_dir / "meta.json").read_text())


@pytest.fixture(scope="session")
def tokenizer(prepared_dir):
    from tinystories_lm.data.tokenizer import BPETokenizer

    return BPETokenizer.load(prepared_dir)


@pytest.fixture(scope="session")
def raw_stories(repo_root) -> list[str]:
    """Every story in the cached raw corpus (~22k), for the roundtrip gate."""
    path = repo_root / "data" / "raw" / "TinyStories-valid.txt"
    if not path.exists():
        pytest.skip(f"{path} not downloaded yet; run scripts/prepare_data.py")
    return parse_stories(path.read_text(encoding="utf-8", errors="replace"))
