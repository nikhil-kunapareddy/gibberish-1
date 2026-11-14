"""Gates 5 and 7: single-batch overfit and resume equivalence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tinystories_lm.model.config import ModelConfig
from tinystories_lm.model.transformer import Transformer
from tinystories_lm.pretrain.optim import build_optimizer, split_param_groups
from tinystories_lm.pretrain.schedule import CosineSchedule
from tinystories_lm.pretrain.train import fit_single_batch, run_pretrain
from tinystories_lm.utils.checkpoint import CheckpointManager, load_checkpoint
from tinystories_lm.utils.config import build_config
from tinystories_lm.utils.seed import seed_everything

# --------------------------------------------------------------------------- #
# GATE 5: single-batch overfit
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_gate5_model_can_memorise_one_batch(prepared_dir, meta):
    """200 steps on one fixed batch must drive the loss below 0.1.

    If this fails, something in the model is silently broken -- a detached
    gradient, a dead residual path, a misaligned label -- and no amount of real
    training will look right.
    """
    from tinystories_lm.data.loader import BatchSampler

    seed_everything(0)
    vocab = meta["tokenizer"]["vocab_size"]
    model = Transformer(
        ModelConfig(vocab_size=vocab, n_layer=2, n_head=4, d_model=128, block_size=64)
    )
    x, y = BatchSampler(prepared_dir, "train", block_size=64, batch_size=4, seed=0).batch()

    losses = fit_single_batch(model, x, y, steps=200, lr=1e-3)

    assert losses[0] > 5.0, f"starting loss {losses[0]:.3f} is implausibly low for a fresh model"
    assert losses[-1] < 0.1, (
        f"failed to memorise one batch: loss {losses[0]:.3f} -> {losses[-1]:.3f} in 200 steps"
    )
    assert losses[-1] < losses[len(losses) // 2], "loss stopped improving halfway through"


def test_gate5_loss_decreases_monotonically_enough_to_be_real(prepared_dir, meta):
    """A cheap version of gate 5 that runs on every `pytest -m 'not slow'`."""
    from tinystories_lm.data.loader import BatchSampler

    seed_everything(0)
    model = Transformer(
        ModelConfig(vocab_size=meta["tokenizer"]["vocab_size"], n_layer=2, n_head=4,
                    d_model=64, block_size=32)
    )
    x, y = BatchSampler(prepared_dir, "train", block_size=32, batch_size=2, seed=0).batch()
    losses = fit_single_batch(model, x, y, steps=40, lr=3e-3)
    assert losses[-1] < 0.6 * losses[0], f"40 steps barely moved the loss: {losses[0]:.3f} -> {losses[-1]:.3f}"


# --------------------------------------------------------------------------- #
# GATE 7: resume equivalence
# --------------------------------------------------------------------------- #


def _pretrain_cfg(prepared_dir: Path, out_dir: Path, **overrides):
    repo = Path(__file__).resolve().parents[1]
    args = [
        f"--pretrain.data_dir={prepared_dir}",
        f"--pretrain.out_dir={out_dir}",
        "--pretrain.device=cpu",
        "--pretrain.amp=false",
        "--pretrain.max_steps=40",
        "--pretrain.batch_size=4",
        "--pretrain.block_size=32",
        "--pretrain.log_interval=1",
        "--pretrain.eval_interval=20",
        "--pretrain.eval_iters=2",
        "--pretrain.ckpt_interval=20",
        "--pretrain.warmup_steps=5",
        "--pretrain.early_ckpt_step=0",
    ]
    args += [f"--{k.replace('__', '.')}={v}" for k, v in overrides.items()]
    return build_config(
        {
            "pretrain": str(repo / "configs/pretrain/smoke.yaml"),
            "model": str(repo / "configs/model/smoke-1m.yaml"),
        },
        args,
    )


def _train_losses(out_dir: Path) -> list[tuple[int, float]]:
    rows = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    seen: dict[int, float] = {}
    for r in rows:
        if r["event"] == "train":
            seen[r["step"]] = r["loss"]  # a re-run of the same step overwrites
    return sorted(seen.items())


@pytest.mark.slow
def test_gate7_resume_matches_an_uninterrupted_run(prepared_dir, meta, tmp_path):
    """20 + 20 resumed steps must reproduce 40 straight steps, loss for loss."""
    vocab = meta["tokenizer"]["vocab_size"]

    straight_dir = tmp_path / "straight"
    cfg = _pretrain_cfg(prepared_dir, straight_dir, model__vocab_size=vocab)
    run_pretrain(cfg.pretrain, cfg.model, quiet=True)

    interrupted_dir = tmp_path / "interrupted"
    cfg_a = _pretrain_cfg(
        prepared_dir, interrupted_dir, model__vocab_size=vocab, pretrain__stop_at_step=20
    )
    run_pretrain(cfg_a.pretrain, cfg_a.model, quiet=True)
    cfg_b = _pretrain_cfg(prepared_dir, interrupted_dir, model__vocab_size=vocab,
                          pretrain__resume="auto")
    run_pretrain(cfg_b.pretrain, cfg_b.model, quiet=True)

    straight, resumed = _train_losses(straight_dir), _train_losses(interrupted_dir)
    assert [s for s, _ in straight] == [s for s, _ in resumed] == list(range(1, 41))
    for (step, a), (_, b) in zip(straight, resumed):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-6), (
            f"step {step}: uninterrupted loss {a:.8f} != resumed loss {b:.8f}. "
            f"Something in the run state is not being checkpointed."
        )

    # ... and the weights themselves, not just the loss trace.
    w_straight = load_checkpoint(straight_dir / "ckpt_last.pt")["model_state"]
    w_resumed = load_checkpoint(interrupted_dir / "ckpt_last.pt")["model_state"]
    for key in w_straight:
        torch.testing.assert_close(w_straight[key], w_resumed[key], rtol=0, atol=0,
                                   msg=f"weight {key} diverged across the resume")


@pytest.mark.slow
def test_resume_restores_optimizer_and_sampler_state(prepared_dir, meta, tmp_path):
    cfg = _pretrain_cfg(prepared_dir, tmp_path / "r", model__vocab_size=meta["tokenizer"]["vocab_size"],
                        pretrain__stop_at_step=20)
    run_pretrain(cfg.pretrain, cfg.model, quiet=True)
    ckpt = load_checkpoint(tmp_path / "r" / "ckpt_last.pt")
    assert ckpt["step"] == 20
    assert ckpt["optimizer_state"]["state"], "optimizer moments were not saved"
    assert ckpt["data_state"]["generator"] is not None
    assert set(ckpt["rng_state"]) >= {"python", "numpy", "torch"}
    assert ckpt["scheduler_state"]["total_steps"] == 40


# --------------------------------------------------------------------------- #
# Checkpoints are self-describing
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_checkpoint_can_be_loaded_with_no_external_knowledge(prepared_dir, meta, tmp_path):
    from tinystories_lm.utils.checkpoint import load_model_from_checkpoint

    cfg = _pretrain_cfg(prepared_dir, tmp_path / "c", model__vocab_size=meta["tokenizer"]["vocab_size"],
                        pretrain__stop_at_step=5)
    result = run_pretrain(cfg.pretrain, cfg.model, quiet=True)

    model, ckpt = load_model_from_checkpoint(result.last_ckpt)
    assert model.cfg.vocab_size == meta["tokenizer"]["vocab_size"]
    assert ckpt["tokenizer"]["hash"] == meta["tokenizer"]["hash"]
    assert ckpt["stage"] == "pretrain" and ckpt["step"] == 5
    with torch.no_grad():
        logits, _ = model(torch.zeros(1, 4, dtype=torch.long))
    assert logits.shape == (1, 4, model.cfg.vocab_size)


@pytest.mark.slow
def test_keep_last_k_rotates_old_checkpoints(prepared_dir, meta, tmp_path):
    out = tmp_path / "rot"
    cfg = _pretrain_cfg(prepared_dir, out, model__vocab_size=meta["tokenizer"]["vocab_size"],
                        pretrain__ckpt_interval=10, pretrain__keep_last_k=2)
    run_pretrain(cfg.pretrain, cfg.model, quiet=True)
    steps = sorted(p.name for p in out.glob("ckpt_step_*.pt"))
    assert len(steps) == 2, f"keep_last_k=2 but found {steps}"
    assert (out / "ckpt_best.pt").exists() and (out / "ckpt_last.pt").exists()


def test_checkpoint_records_best_val_loss_including_its_own(tmp_path):
    """A checkpoint's `best_val_loss` must include the val loss measured at that step.

    Regression: `save` used to write the file *before* updating the running best,
    so every checkpoint carried the previous best. On resume that stale value is
    restored, and a later, worse checkpoint could then overwrite ckpt_best.pt.
    """
    cfg = ModelConfig(vocab_size=32, n_layer=1, n_head=2, d_model=16, block_size=8)
    model = Transformer(cfg)
    ckpts = CheckpointManager(tmp_path, keep_last_k=10)

    for step, val in [(1, 2.0), (2, None), (3, 1.5), (4, 1.8)]:
        ckpts.save(step, val_loss=val, model=model, model_config=cfg)

    best = load_checkpoint(ckpts.best_path)
    assert best["step"] == 3 and best["best_val_loss"] == 1.5
    assert load_checkpoint(ckpts.step_path(1))["best_val_loss"] == 2.0
    assert load_checkpoint(ckpts.last_path)["best_val_loss"] == 1.5

    # Resume the way the training loops do, then save a checkpoint worse than
    # the true best (1.5) but better than the old stale value (2.0).
    resumed = CheckpointManager(tmp_path, keep_last_k=10)
    resumed.best_val_loss = load_checkpoint(resumed.last_path)["best_val_loss"]
    resumed.save(5, val_loss=1.6, model=model, model_config=cfg)
    assert load_checkpoint(resumed.best_path)["step"] == 3, "a worse checkpoint replaced ckpt_best.pt"


def test_stage_rejects_a_vocab_mismatch(prepared_dir, meta, tmp_path):
    cfg = _pretrain_cfg(prepared_dir, tmp_path / "bad", model__vocab_size=1234)
    with pytest.raises(ValueError, match="vocab_size"):
        run_pretrain(cfg.pretrain, cfg.model, quiet=True)


def test_stage_rejects_an_unprepared_data_dir(tmp_path):
    cfg = _pretrain_cfg(tmp_path / "nothing-here", tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        run_pretrain(cfg.pretrain, cfg.model, quiet=True)


# --------------------------------------------------------------------------- #
# Optimizer and schedule
# --------------------------------------------------------------------------- #


def test_weight_decay_skips_norms_and_embeddings():
    model = Transformer(ModelConfig(vocab_size=256, n_layer=2, n_head=2, d_model=32, block_size=16))
    decay, no_decay = split_param_groups(model, 0.1)
    decayed = {id(p) for p in decay["params"]}
    for name, param in model.named_parameters():
        wants_decay = name.endswith(("qkv_proj.weight", "out_proj.weight", "gate_proj.weight",
                                     "up_proj.weight", "down_proj.weight"))
        assert (id(param) in decayed) == wants_decay, f"{name} is in the wrong decay group"
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0


def test_optimizer_covers_every_parameter_exactly_once():
    model = Transformer(ModelConfig(vocab_size=256, n_layer=2, n_head=2, d_model=32, block_size=16))
    opt = build_optimizer(model, lr=1e-3)
    in_opt = [id(p) for g in opt.param_groups for p in g["params"]]
    unique_model_params = {id(p) for p in model.parameters()}
    assert len(in_opt) == len(set(in_opt)) == len(unique_model_params)


def test_schedule_warms_up_then_decays_to_the_floor():
    s = CosineSchedule(base_lr=1e-3, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert s.lr_at(0) == pytest.approx(1e-4)
    assert s.lr_at(9) == pytest.approx(1e-3)  # warmup ends at the peak
    assert s.lr_at(10) == pytest.approx(1e-3)
    assert s.lr_at(55) == pytest.approx(s.min_lr + 0.5 * (s.base_lr - s.min_lr), rel=0.02)
    assert s.lr_at(100) == pytest.approx(1e-4)
    assert s.lr_at(10_000) == pytest.approx(1e-4), "LR must not go negative past the horizon"
    warm = [s.lr_at(i) for i in range(10)]
    assert warm == sorted(warm)
    decay = [s.lr_at(i) for i in range(10, 100)]
    assert decay == sorted(decay, reverse=True)


def test_schedule_survives_a_roundtrip():
    s = CosineSchedule(1e-3, 10, 100, 0.1)
    assert CosineSchedule.from_state_dict(s.state_dict()).lr_at(42) == s.lr_at(42)


def test_schedule_rejects_impossible_warmup():
    with pytest.raises(ValueError, match="warmup_steps"):
        CosineSchedule(1e-3, warmup_steps=200, total_steps=100)


def test_grad_accumulation_matches_one_big_batch():
    """Gradient accumulation must average, not sum -- otherwise the effective LR
    silently scales with grad_accum_steps."""
    seed_everything(0)
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_head=2, d_model=32, block_size=16)
    model = Transformer(cfg)
    x = torch.randint(0, 256, (8, 16))
    y = torch.randint(0, 256, (8, 16))

    model.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    big = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    for i in range(2):
        _, loss = model(x[i * 4 : (i + 1) * 4], y[i * 4 : (i + 1) * 4])
        (loss / 2).backward()
    accumulated = [p.grad.clone() for p in model.parameters()]

    for a, b in zip(big, accumulated):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
