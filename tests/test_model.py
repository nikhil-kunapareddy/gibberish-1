"""Gates 4 and 6: untrained-loss sanity and causal mask integrity."""

from __future__ import annotations

import math

import pytest
import torch

from tinystories_lm.model.config import ModelConfig
from tinystories_lm.model.transformer import IGNORE_INDEX, Transformer
from tinystories_lm.utils.seed import seed_everything


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=512, n_layer=2, n_head=4, d_model=64, block_size=64, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


@pytest.fixture
def model() -> Transformer:
    seed_everything(0)
    m = Transformer(tiny_cfg())
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# GATE 4: untrained loss ~ ln(vocab_size)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("vocab_size", [512, 4096, 8192])
def test_gate4_untrained_loss_is_within_5pct_of_ln_vocab(vocab_size):
    """A fresh model predicts a near-uniform distribution: loss ~= ln(V).

    Too low means the labels are aligned with the inputs somewhere they should
    not be; far off in either direction means the init is wrong.
    """
    seed_everything(0)
    m = Transformer(tiny_cfg(vocab_size=vocab_size))
    m.eval()
    x = torch.randint(0, vocab_size, (8, 64))
    y = torch.randint(0, vocab_size, (8, 64))
    with torch.no_grad():
        _, loss = m(x, y)
    expected = math.log(vocab_size)
    rel = abs(loss.item() - expected) / expected
    assert rel < 0.05, f"untrained loss {loss.item():.4f} vs ln(V)={expected:.4f} (off by {rel:.1%})"


def test_gate4_holds_on_real_token_statistics(prepared_dir, meta):
    """Same check on real data, where token frequencies are far from uniform."""
    from tinystories_lm.data.loader import BatchSampler

    seed_everything(0)
    vocab = meta["tokenizer"]["vocab_size"]
    m = Transformer(tiny_cfg(vocab_size=vocab, block_size=128))
    m.eval()
    sampler = BatchSampler(prepared_dir, "train", block_size=128, batch_size=8, seed=0)
    x, y = sampler.batch()
    with torch.no_grad():
        _, loss = m(x, y)
    expected = math.log(vocab)
    assert abs(loss.item() - expected) / expected < 0.05


# --------------------------------------------------------------------------- #
# GATE 6: causal mask integrity
# --------------------------------------------------------------------------- #


def test_gate6_perturbing_token_t_cannot_change_logits_before_t(model):
    """Information must never flow backwards.

    A leaky mask shows up as suspiciously good training loss and is otherwise
    invisible, so check it directly: change one input token and assert every
    earlier position's logits are bit-identical.
    """
    torch.manual_seed(0)
    x = torch.randint(0, model.cfg.vocab_size, (2, 32))
    with torch.no_grad():
        base, _ = model(x)

    for t in (1, 7, 16, 31):
        perturbed = x.clone()
        perturbed[:, t] = (perturbed[:, t] + 1) % model.cfg.vocab_size
        with torch.no_grad():
            after, _ = model(perturbed)
        assert torch.equal(base[:, :t], after[:, :t]), (
            f"perturbing position {t} changed logits at earlier positions: the mask leaks"
        )
        assert not torch.equal(base[:, t], after[:, t]), (
            f"perturbing position {t} did not change its own logits: the input is being ignored"
        )


def test_gate6_prefix_invariance(model):
    """Logits for a prefix must not depend on what follows it."""
    torch.manual_seed(1)
    full = torch.randint(0, model.cfg.vocab_size, (1, 24))
    with torch.no_grad():
        full_logits, _ = model(full)
        prefix_logits, _ = model(full[:, :10])
    torch.testing.assert_close(full_logits[:, :10], prefix_logits, rtol=1e-4, atol=1e-5)


def test_gate6_attention_mask_helper_is_lower_triangular():
    from tinystories_lm.model.attention import causal_mask

    mask = causal_mask(4, 4, torch.device("cpu"))
    assert torch.equal(mask, torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    # A 2-token window at the end of a 6-token history may see all 4 prior
    # tokens plus itself, but not the token after it.
    windowed = causal_mask(2, 6, torch.device("cpu"))
    assert windowed.tolist() == [
        [True, True, True, True, True, False],
        [True, True, True, True, True, True],
    ]
    assert causal_mask(1, 9, torch.device("cpu")) is None  # decode: attend to everything


# --------------------------------------------------------------------------- #
# Architecture properties
# --------------------------------------------------------------------------- #


def test_no_bias_terms_anywhere(model):
    biased = [n for n, _ in model.named_parameters() if n.endswith("bias")]
    assert biased == [], f"expected a bias-free model, found {biased}"


def test_embeddings_are_tied(model):
    assert model.lm_head.weight is model.tok_emb.weight


def test_untied_embeddings_are_supported():
    m = Transformer(tiny_cfg(tie_embeddings=False))
    assert m.lm_head.weight is not m.tok_emb.weight
    assert m.num_params(True) - m.num_params(False) == 2 * 512 * 64


def test_num_params_excludes_embeddings_when_asked(model):
    b = model.param_breakdown()
    assert model.num_params(include_embeddings=True) == b["total"]
    assert model.num_params(include_embeddings=False) == b["non_embedding"]
    assert b["embedding"] == model.cfg.vocab_size * model.cfg.d_model  # tied: counted once


def test_33m_config_lands_near_33m_non_embedding_params(repo_root):
    import yaml

    cfg = ModelConfig(**yaml.safe_load((repo_root / "configs/model/tiny-33m.yaml").read_text()))
    m = Transformer(cfg)
    non_emb = m.num_params(include_embeddings=False)
    assert 30e6 < non_emb < 36e6, f"tiny-33m has {non_emb / 1e6:.1f}M non-embedding params"
    # The point of the 8k vocab: the embedding must not dominate the model.
    assert m.param_breakdown()["embedding"] < 0.2 * m.num_params(True)


def test_rope_has_no_learned_position_parameters(model):
    assert not any("pos" in n or "wpe" in n for n, _ in model.named_parameters())
    assert "rope_cos" not in model.state_dict(), "rope cache must not be checkpointed"


def test_ignore_index_positions_do_not_affect_the_loss(model):
    """The mechanism SFT masking relies on (also checked end-to-end in gate 9)."""
    torch.manual_seed(0)
    x = torch.randint(0, model.cfg.vocab_size, (2, 16))
    y = torch.randint(0, model.cfg.vocab_size, (2, 16))
    y_masked = y.clone()
    y_masked[:, :8] = IGNORE_INDEX
    with torch.no_grad():
        _, masked_loss = model(x, y_masked)
        logits, _ = model(x)
        manual = torch.nn.functional.cross_entropy(
            logits[:, 8:].reshape(-1, model.cfg.vocab_size).float(), y[:, 8:].reshape(-1)
        )
    torch.testing.assert_close(masked_loss, manual, rtol=1e-5, atol=1e-6)


def test_forward_rejects_sequences_longer_than_block_size(model):
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, model.cfg.block_size + 1, dtype=torch.long))


def test_last_token_only_matches_the_full_forward(model):
    torch.manual_seed(0)
    x = torch.randint(0, model.cfg.vocab_size, (2, 12))
    with torch.no_grad():
        full, _ = model(x)
        last, _ = model(x, last_token_only=True)
    assert last.shape == (2, 1, model.cfg.vocab_size)
    torch.testing.assert_close(full[:, -1:], last)


def test_dropout_is_active_only_in_training_mode():
    seed_everything(0)
    m = Transformer(tiny_cfg(dropout=0.3))
    x = torch.randint(0, 512, (2, 16))
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x)[0], m(x)[0])
    m.train()
    with torch.no_grad():
        assert not torch.equal(m(x)[0], m(x)[0])
