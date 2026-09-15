"""Gate 8: KV-cache equivalence, plus the sampling filters."""

from __future__ import annotations

import pytest
import torch

from tinystories_lm.inference.generate import (
    generate,
    generate_text,
    profile_generation,
)
from tinystories_lm.inference.sampling import (
    apply_temperature,
    sample_from_logits,
    top_k_filter,
    top_p_filter,
)
from tinystories_lm.model.config import ModelConfig
from tinystories_lm.model.transformer import Transformer
from tinystories_lm.utils.seed import seed_everything


@pytest.fixture(scope="module")
def model() -> Transformer:
    seed_everything(0)
    m = Transformer(ModelConfig(vocab_size=256, n_layer=3, n_head=4, d_model=64, block_size=64))
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# GATE 8: KV cache equivalence
# --------------------------------------------------------------------------- #


def test_gate8_cached_and_uncached_greedy_decoding_are_identical(model):
    """50 greedy tokens from the same prompt, cache on vs cache off."""
    torch.manual_seed(0)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 8))
    cached = generate(model, prompt, 50, greedy=True, use_cache=True)
    uncached = generate(model, prompt, 50, greedy=True, use_cache=False)
    assert torch.equal(cached, uncached), (
        f"cache changed the output at position "
        f"{int((cached != uncached).float().argmax())}: "
        f"{cached.tolist()[0][-10:]} vs {uncached.tolist()[0][-10:]}"
    )
    assert cached.shape == (1, 58)


def test_gate8_holds_for_batched_prompts(model):
    torch.manual_seed(1)
    prompt = torch.randint(0, model.cfg.vocab_size, (4, 5))
    assert torch.equal(
        generate(model, prompt, 30, greedy=True, use_cache=True),
        generate(model, prompt, 30, greedy=True, use_cache=False),
    )


@pytest.mark.parametrize("prompt_len,new_tokens", [(50, 40), (8, 200), (63, 130)])
def test_gate8_holds_across_the_block_size_boundary(model, prompt_len, new_tokens):
    """Both paths must slide the context window the same way.

    block_size is 64 here, so each case crosses the boundary at least once and
    the 200-token case rebuilds the cache several times.
    """
    torch.manual_seed(2)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len))
    cached = generate(model, prompt, new_tokens, greedy=True, use_cache=True)
    uncached = generate(model, prompt, new_tokens, greedy=True, use_cache=False)
    assert cached.shape == uncached.shape == (1, prompt_len + new_tokens)
    assert torch.equal(cached, uncached), (
        f"diverged at position {int((cached != uncached).float().argmax())} "
        f"of {cached.shape[1]}"
    )


def test_generation_longer_than_block_size_returns_full_history(model):
    """Cropping the context must not crop the returned sequence."""
    prompt = torch.randint(0, model.cfg.vocab_size, (2, 10))
    out = generate(model, prompt, 120, greedy=True, use_cache=True)
    assert out.shape == (2, 130)
    assert torch.equal(out[:, :10], prompt)


def test_cached_logits_match_a_full_forward_pass(model):
    """The cache must reproduce the full-sequence logits, not merely 'look sane'."""
    torch.manual_seed(3)
    x = torch.randint(0, model.cfg.vocab_size, (2, 16))
    with torch.no_grad():
        full, _ = model(x)
        cache = model.new_kv_cache(2, 32)
        incremental = []
        for t in range(x.size(1)):
            logits, _ = model(x[:, t : t + 1], cache=cache)
            incremental.append(logits)
    torch.testing.assert_close(torch.cat(incremental, dim=1), full, rtol=1e-4, atol=1e-5)


def test_cache_prefill_then_decode_matches_full_forward(model):
    torch.manual_seed(4)
    x = torch.randint(0, model.cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full, _ = model(x)
        cache = model.new_kv_cache(1, 32)
        model(x[:, :8], cache=cache)          # prefill
        logits, _ = model(x[:, 8:], cache=cache)  # decode the rest in one go
    torch.testing.assert_close(logits, full[:, 8:], rtol=1e-4, atol=1e-5)


def test_cache_overflow_is_an_error_not_silent_corruption(model):
    cache = model.new_kv_cache(1, 8)
    with pytest.raises(ValueError, match="overflow"):
        model(torch.zeros(1, 9, dtype=torch.long), cache=cache)


def test_cache_position_advances_once_per_forward_not_once_per_layer(model):
    cache = model.new_kv_cache(1, 32)
    model(torch.zeros(1, 5, dtype=torch.long), cache=cache)
    assert cache.pos == 5, f"cache.pos={cache.pos} after one 5-token forward"
    model(torch.zeros(1, 1, dtype=torch.long), cache=cache)
    assert cache.pos == 6


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_greedy_is_deterministic_and_picks_the_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.3], [2.0, 0.0, 1.0]])
    assert sample_from_logits(logits, greedy=True).flatten().tolist() == [1, 0]
    assert sample_from_logits(logits, temperature=0.0).flatten().tolist() == [1, 0]


def test_temperature_scales_logits():
    logits = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(apply_temperature(logits, 0.5), torch.tensor([[2.0, 4.0]]))
    with pytest.raises(ValueError):
        apply_temperature(logits, -1.0)


def test_top_k_keeps_exactly_k_tokens():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
    filtered = top_k_filter(logits, 2)
    assert torch.isinf(filtered).sum() == 2
    assert filtered.flatten().tolist()[1] == 3.0 and filtered.flatten().tolist()[2] == 2.0
    assert torch.equal(top_k_filter(logits, 0), logits), "k=0 must disable the filter"


def test_top_p_keeps_the_nucleus():
    # softmax([3, 2, 1, 0]) ~ [0.644, 0.237, 0.087, 0.032]
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    kept = ~torch.isinf(top_p_filter(logits, 0.8))
    assert kept.flatten().tolist() == [True, True, False, False]
    kept = ~torch.isinf(top_p_filter(logits, 0.5))
    assert kept.flatten().tolist() == [True, False, False, False], "top-1 must always survive"
    assert torch.equal(top_p_filter(logits, 1.0), logits), "p=1 must disable the filter"


def test_sampling_respects_the_filters():
    """A masked token must never be sampled, however many draws you take."""
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    seed_everything(0)
    draws = {int(sample_from_logits(logits, temperature=1.0, top_k=2)) for _ in range(200)}
    assert draws <= {0, 1}, f"top_k=2 sampled outside the top 2: {draws}"


def test_sampling_is_reproducible_under_a_seed():
    logits = torch.randn(1, 50)
    seed_everything(7)
    a = [int(sample_from_logits(logits, temperature=1.0)) for _ in range(10)]
    seed_everything(7)
    b = [int(sample_from_logits(logits, temperature=1.0)) for _ in range(10)]
    assert a == b


def test_sample_rejects_wrong_shaped_logits():
    with pytest.raises(ValueError, match="B, V"):
        sample_from_logits(torch.randn(2, 3, 4))


# --------------------------------------------------------------------------- #
# Text API
# --------------------------------------------------------------------------- #


def test_generate_text_returns_decoded_completions(tokenizer, prepared_dir, meta):
    seed_everything(0)
    model = Transformer(
        ModelConfig(vocab_size=meta["tokenizer"]["vocab_size"], n_layer=2, n_head=2,
                    d_model=32, block_size=64)
    )
    outs = generate_text(model, tokenizer, "Once upon a time", max_new_tokens=16,
                         num_samples=2, greedy=False)
    assert len(outs) == 2
    for g in outs:
        assert g.text.startswith("Once upon a time")
        assert g.text == g.prompt + g.completion
        assert 0 < g.n_new_tokens <= 16


def test_generate_text_stops_at_eot(tokenizer, meta):
    """EOT must terminate the completion rather than appear inside it."""
    seed_everything(0)
    model = Transformer(
        ModelConfig(vocab_size=meta["tokenizer"]["vocab_size"], n_layer=2, n_head=2,
                    d_model=32, block_size=64)
    )
    # Force EOT by making it the argmax everywhere.
    with torch.no_grad():
        model.lm_head.weight[tokenizer.eot_id] += 50.0
    out = generate_text(model, tokenizer, "Hi", max_new_tokens=20, greedy=True, stop_at_eot=True)[0]
    assert out.completion == ""
    assert out.n_new_tokens == 0


def test_profile_reports_both_paths_and_the_same_text(tokenizer, meta):
    seed_everything(0)
    model = Transformer(
        ModelConfig(vocab_size=meta["tokenizer"]["vocab_size"], n_layer=2, n_head=2,
                    d_model=32, block_size=128)
    )
    stats = profile_generation(model, tokenizer, "Once upon a time", max_new_tokens=32)
    assert stats["identical_output"], "profiling the two paths produced different text"
    assert stats["with_cache"]["tokens"] == 32
    assert stats["no_cache"]["tokens_per_sec"] > 0 and stats["with_cache"]["tokens_per_sec"] > 0
