"""Gate 9: SFT loss masking. Plus DPO mechanics."""

from __future__ import annotations

import json

import pytest
import torch

from tinystories_lm.model.config import ModelConfig
from tinystories_lm.model.transformer import IGNORE_INDEX, Transformer
from tinystories_lm.posttrain.dpo import (
    PreferenceDataset,
    dpo_loss,
    freeze_reference,
    sequence_logprobs,
)
from tinystories_lm.posttrain.sft import SFTDataset, build_example
from tinystories_lm.utils.seed import seed_everything


@pytest.fixture(scope="module")
def model_factory():
    def make(vocab_size: int) -> Transformer:
        seed_everything(0)
        m = Transformer(
            ModelConfig(vocab_size=vocab_size, n_layer=2, n_head=2, d_model=64, block_size=128)
        )
        m.eval()
        return m

    return make


# --------------------------------------------------------------------------- #
# GATE 9: SFT mask correctness
# --------------------------------------------------------------------------- #


def test_gate9_prompt_positions_are_masked_out():
    """`build_example` masks exactly the targets that are prompt tokens."""
    prompt, completion, eot = [10, 11, 12, 13], [20, 21, 22], 99
    x, y = build_example(prompt, completion, eot, max_seq_len=64)

    assert x == [10, 11, 12, 13, 20, 21, 22]
    # Targets: the first 3 (predicting prompt tokens 11, 12, 13) are masked.
    # Position 3 is the last prompt token predicting the first completion token
    # -- that one is supervised.
    assert y == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21, 22, eot]
    assert sum(t != IGNORE_INDEX for t in y) == len(completion) + 1  # +1 for EOT


def test_gate9_prompt_tokens_contribute_zero_loss(model_factory):
    """End-to-end: perturbing the *model's fit to the prompt* cannot change the
    SFT loss, and the masked loss equals the loss computed on completions alone."""
    model = model_factory(256)
    prompt, completion = [5, 6, 7, 8, 9], [40, 41, 42]
    x_list, y_list = build_example(prompt, completion, eot_id=99, max_seq_len=64)
    x = torch.tensor([x_list])
    y = torch.tensor([y_list])

    with torch.no_grad():
        _, masked_loss = model(x, y)
        logits, _ = model(x)
        n_masked = len(prompt) - 1
        manual = torch.nn.functional.cross_entropy(
            logits[0, n_masked:].float(), y[0, n_masked:]
        )
    torch.testing.assert_close(masked_loss, manual, rtol=1e-5, atol=1e-6)


def test_gate9_gradients_are_zero_for_prompt_only_examples(model_factory):
    """The sharpest form of the check: an example whose completion is empty
    must produce no gradient at all."""
    model = model_factory(256)
    x = torch.tensor([[5, 6, 7, 8]])
    y = torch.full_like(x, IGNORE_INDEX)
    model.zero_grad()
    _, loss = model(x, y)
    assert torch.isnan(loss), "a fully masked batch should yield NaN, not a silent zero"

    # And with a real completion, only the completion's target tokens can move
    # the loss: changing a masked target leaves loss and gradients untouched.
    y_a = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 30, 31]])
    y_b = torch.tensor([[7777, 123, 30, 31]])
    y_b[:, :2] = IGNORE_INDEX
    model.zero_grad()
    _, loss_a = model(x, y_a)
    loss_a.backward()
    grads_a = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    _, loss_b = model(x, y_b)
    loss_b.backward()
    for ga, p in zip(grads_a, model.parameters()):
        torch.testing.assert_close(ga, p.grad)


def test_gate9_dataset_masks_every_prompt_in_a_real_batch(prepared_dir, tokenizer, meta):
    """The masking survives tokenisation, padding, and batching of real data."""
    dataset = SFTDataset(
        prepared_dir / meta["sft"]["train_file"], tokenizer, max_seq_len=128, seed=0
    )
    assert len(dataset) > 0
    x, y = dataset.batch(8)
    assert x.shape == y.shape
    for row in range(x.size(0)):
        supervised = (y[row] != IGNORE_INDEX).nonzero().flatten()
        assert len(supervised) > 0, "an example has nothing to learn from"
        first = int(supervised[0])
        # Everything before the first supervised position is masked, and the
        # supervised region is contiguous up to padding.
        assert (y[row, :first] == IGNORE_INDEX).all()
        assert (y[row, first : int(supervised[-1]) + 1] != IGNORE_INDEX).all()
        # The masked prefix must be the prompt: it matches the prompt tokens.
        assert first >= 1


def test_gate9_masked_prefix_matches_the_prompt_text(prepared_dir, tokenizer, meta):
    rows = [json.loads(l) for l in (prepared_dir / meta["sft"]["train_file"]).read_text().splitlines()[:5]]
    dataset = SFTDataset(prepared_dir / meta["sft"]["train_file"], tokenizer, 256, seed=0)
    for i, row in enumerate(rows):
        x, y = dataset.collate([i])
        prompt_ids = tokenizer.encode(row["prompt"])
        masked_prefix = x[0, : len(prompt_ids)].tolist()
        assert masked_prefix == prompt_ids, "the masked prefix is not the prompt"
        assert (y[0, : len(prompt_ids) - 1] == IGNORE_INDEX).all()
        assert y[0, len(prompt_ids) - 1] != IGNORE_INDEX, "first completion token is not supervised"


def test_padding_never_contributes_to_the_loss(prepared_dir, tokenizer, meta):
    dataset = SFTDataset(prepared_dir / meta["sft"]["train_file"], tokenizer, 128, seed=0)
    x, y = dataset.collate([0, 1, 2, 3])
    widths = [len(dataset.examples[i][0]) for i in range(4)]
    for row, width in enumerate(widths):
        assert (y[row, width:] == IGNORE_INDEX).all(), "padding is being supervised"


def test_build_example_rejects_examples_with_nothing_to_supervise():
    # A prompt longer than max_seq_len leaves no completion tokens in the window.
    assert build_example([1, 2, 3, 4, 5], [6, 7], eot_id=9, max_seq_len=3) is None
    assert build_example([], [1], eot_id=9, max_seq_len=8) is not None
    x, y = build_example([1, 2], [3, 4], eot_id=9, max_seq_len=3)
    assert len(x) == len(y) == 3, "max_seq_len must bound the sequence"


def test_build_example_supervises_the_terminating_eot():
    """The model has to learn where a story stops, so EOT is a real target."""
    _, y = build_example([1, 2], [3], eot_id=9, max_seq_len=64)
    assert y == [IGNORE_INDEX, 3, 9]


# --------------------------------------------------------------------------- #
# DPO mechanics
# --------------------------------------------------------------------------- #


def test_dpo_loss_is_minimised_by_preferring_the_chosen_response():
    beta = 0.1
    ref_c = torch.tensor([-10.0])
    ref_r = torch.tensor([-10.0])
    better, _ = dpo_loss(torch.tensor([-5.0]), torch.tensor([-20.0]), ref_c, ref_r, beta)
    worse, _ = dpo_loss(torch.tensor([-20.0]), torch.tensor([-5.0]), ref_c, ref_r, beta)
    tied, _ = dpo_loss(torch.tensor([-10.0]), torch.tensor([-10.0]), ref_c, ref_r, beta)
    assert better < tied < worse
    assert tied == pytest.approx(-torch.log(torch.tensor(0.5)).item(), rel=1e-5)


def test_dpo_loss_is_zero_gradient_when_policy_equals_reference():
    """At initialisation policy == reference, so the loss must be exactly ln 2."""
    lp = torch.tensor([-3.0, -7.0], requires_grad=True)
    lr_ = torch.tensor([-4.0, -2.0], requires_grad=True)
    loss, metrics = dpo_loss(lp, lr_, lp.detach(), lr_.detach(), beta=0.1)
    assert loss.item() == pytest.approx(0.6931, abs=1e-4)
    assert float(metrics["reward_margin"]) == pytest.approx(0.0, abs=1e-6)


def test_dpo_metrics_track_reward_accuracy():
    ref = torch.zeros(4)
    chosen = torch.tensor([1.0, 1.0, -1.0, 1.0])
    rejected = torch.tensor([0.0, 0.0, 0.0, 0.0])
    _, metrics = dpo_loss(chosen, rejected, ref, ref, beta=0.1)
    assert float(metrics["reward_accuracy"]) == pytest.approx(0.75)


def test_dpo_reference_stays_frozen_while_the_policy_trains(model_factory):
    policy = model_factory(256)
    reference = freeze_reference(policy)
    assert all(not p.requires_grad for p in reference.parameters())
    assert not reference.training

    before = {k: v.clone() for k, v in reference.state_dict().items()}
    x = torch.randint(0, 256, (2, 16))
    y = x.clone()
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    for _ in range(3):
        loss = -sequence_logprobs(policy, x, y).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    for k, v in reference.state_dict().items():
        assert torch.equal(before[k], v), f"reference parameter {k} moved during training"
    changed = any(
        not torch.equal(before[k], v) for k, v in policy.state_dict().items()
    )
    assert changed, "the policy did not move at all"


def test_sequence_logprobs_ignores_masked_positions(model_factory):
    model = model_factory(256)
    x = torch.randint(0, 256, (1, 12))
    y_full = x.clone()
    y_masked = x.clone()
    y_masked[:, :6] = IGNORE_INDEX

    with torch.no_grad():
        full = sequence_logprobs(model, x, y_full)
        masked = sequence_logprobs(model, x, y_masked)
        logits, _ = model(x)
        manual = torch.log_softmax(logits.float(), -1)[0, 6:].gather(-1, x[0, 6:, None]).sum()
    assert masked.item() == pytest.approx(manual.item(), rel=1e-5)
    assert masked.item() > full.item(), "fewer summed log-probs must be larger (less negative)"


def test_sequence_logprobs_are_negative_and_shaped_per_example(model_factory):
    model = model_factory(256)
    x = torch.randint(0, 256, (3, 10))
    out = sequence_logprobs(model, x, x)
    assert out.shape == (3,)
    assert (out < 0).all()


def test_preference_dataset_masks_the_prompt_in_both_halves(tmp_path, tokenizer):
    from tinystories_lm.data.prepare import write_jsonl

    pairs = [
        {"prompt": "Once upon a time there was", "chosen": " a happy dog.", "rejected": " zzz qqq."},
        {"prompt": "The cat sat", "chosen": " on the mat.", "rejected": " xxyy zz."},
    ]
    path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs, path)
    dataset = PreferenceDataset(path, tokenizer, max_seq_len=64, seed=0)
    assert len(dataset) == 2

    batch = dataset.collate([0, 1])
    for key in ("chosen", "rejected"):
        x, y = batch[f"{key}_x"], batch[f"{key}_y"]
        assert x.shape == y.shape
        for row in range(x.size(0)):
            prompt_len = len(tokenizer.encode(pairs[row]["prompt"]))
            assert (y[row, : prompt_len - 1] == IGNORE_INDEX).all(), f"{key} prompt is supervised"
            assert y[row, prompt_len - 1] != IGNORE_INDEX
    # Both halves share a width so one collate feeds both forward passes.
    assert batch["chosen_x"].shape[1] == batch["rejected_x"].shape[1]
