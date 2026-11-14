"""The pretraining loop.

Hand-written on purpose (no Trainer, no Lightning, no Accelerate): everything
that happens to a gradient is visible in `run_pretrain` below.

Input contract:  a prepared data dir (train.bin / val.bin / meta.json).
Output contract: `<out_dir>/ckpt_{last,best,early}.pt`, `metrics.jsonl`,
                 `config.yaml`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.loader import BatchSampler
from ..data.prepare import validate_data_dir
from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..utils.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    restore_training_state,
)
from ..utils.device import autocast_ctx, resolve_device, synchronize, use_amp
from ..utils.logging import RunLogger
from ..utils.seed import seed_everything
from .optim import build_optimizer, describe_param_groups
from .schedule import CosineSchedule


@dataclass
class StageResult:
    """What a training stage hands to the next one."""

    stage: str
    out_dir: Path
    last_ckpt: Path
    best_ckpt: Path
    steps: int
    final_train_loss: float
    best_val_loss: float
    tokens_seen: int
    elapsed_sec: float


def check_model_data_compat(model_cfg: ModelConfig, meta: dict, block_size: int) -> None:
    """Validate the stage's input contract before a single tensor is allocated."""
    tok_vocab = meta["tokenizer"]["vocab_size"]
    if model_cfg.vocab_size != tok_vocab:
        raise ValueError(
            f"model vocab_size={model_cfg.vocab_size} but the prepared data was tokenized "
            f"with vocab_size={tok_vocab}.\n"
            f"  Fix by matching the configs, e.g. --model.vocab_size={tok_vocab}"
        )
    if block_size > model_cfg.block_size:
        raise ValueError(
            f"training block_size={block_size} exceeds the model's block_size="
            f"{model_cfg.block_size} (the RoPE cache is built for the latter)"
        )


@torch.no_grad()
def evaluate(model, sampler: BatchSampler, iters: int, device_info, amp_enabled: bool) -> float:
    """Mean loss over `iters` fixed validation batches."""
    was_training = model.training
    model.eval()
    losses = []
    for x, y in sampler.eval_batches(iters, seed=0):
        with autocast_ctx(device_info, amp_enabled):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training)
    return float(sum(losses) / max(1, len(losses)))


def fit_single_batch(
    model: Transformer,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    log_every: int = 0,
) -> list[float]:
    """Train one fixed batch to memorisation (gate 5).

    Kept in the training module rather than the test file because it is the
    first thing to run when a model change makes the loss curve look wrong: if
    a model cannot overfit 4x64 tokens, nothing downstream is worth debugging.
    """
    model.train()
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    schedule = CosineSchedule(base_lr=lr, warmup_steps=max(1, steps // 20), total_steps=steps)
    losses = []
    for step in range(steps):
        schedule.apply(optimizer, step)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"  overfit step {step:4d} loss {loss.item():.4f}")
    return losses


def run_pretrain(
    cfg,
    model_cfg: ModelConfig,
    full_config: dict | None = None,
    quiet: bool = False,
) -> StageResult:
    t_start = time.time()
    meta = validate_data_dir(cfg.data_dir)
    check_model_data_compat(model_cfg, meta, cfg.block_size)

    seed_everything(cfg.seed)
    device_info = resolve_device(cfg.device)
    amp_enabled = use_amp(device_info, cfg.amp)

    out_dir = Path(cfg.out_dir)
    logger = RunLogger(out_dir, cfg.name, cfg.wandb, cfg.wandb_project, full_config)
    logger.info(f"[pretrain] device={device_info} amp={amp_enabled} data={cfg.data_dir}")

    train_sampler = BatchSampler(
        cfg.data_dir, "train", cfg.block_size, cfg.batch_size, device_info.device, cfg.seed
    )
    val_sampler = BatchSampler(
        cfg.data_dir, "val", cfg.block_size, cfg.batch_size, device_info.device, cfg.seed + 1
    )

    model = Transformer(model_cfg).to(device_info.device)
    logger.info(model.describe())
    optimizer = build_optimizer(
        model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        device_type=device_info.type,
    )
    logger.info("[pretrain] param groups:\n" + describe_param_groups(model, cfg.weight_decay))
    schedule = CosineSchedule(cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
    ckpts = CheckpointManager(out_dir, cfg.keep_last_k, stage="pretrain")

    start_step = 0
    resume_path = ckpts.resolve_resume(cfg.resume)
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location=device_info.device)
        start_step = restore_training_state(ckpt, model=model, optimizer=optimizer)
        if ckpt.get("data_state"):
            train_sampler.load_state_dict(ckpt["data_state"])
        ckpts.best_val_loss = ckpt.get("best_val_loss") or float("inf")
        logger.info(f"[pretrain] resumed from {resume_path} at step {start_step}")

    raw_model = model
    if cfg.compile:
        logger.info("[pretrain] torch.compile: warming up (first steps will be slow)")
        model = torch.compile(model)

    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps
    ckpt_common = dict(
        model_config=model_cfg,
        tokenizer=meta["tokenizer"],
        data_meta={"data_dir": str(cfg.data_dir), "splits": {
            k: {kk: vv for kk, vv in v.items() if kk != "story_hashes"}
            for k, v in meta["splits"].items()
        }},
        config=full_config or {},
    )

    model.train()
    last_loss = float("nan")
    # The LR schedule always spans max_steps; stop_at_step only decides when to
    # walk away, so an interrupted run resumes onto the same curve.
    final_step = min(cfg.stop_at_step, cfg.max_steps) if cfg.stop_at_step else cfg.max_steps
    t_step = time.time()
    for step in range(start_step, final_step):
        lr = schedule.apply(optimizer, step)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_sampler.batch()
            with autocast_ctx(device_info, amp_enabled):
                _, loss = model(x, y)
            # Scale so the gradient matches one big batch, not the sum of micros.
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item() / cfg.grad_accum_steps

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()
        last_loss = accum_loss

        if (step + 1) % cfg.log_interval == 0 or step == start_step:
            synchronize(device_info)
            dt = time.time() - t_step
            n = cfg.log_interval if (step + 1) % cfg.log_interval == 0 else 1
            logger.metrics(
                "train",
                step=step + 1,
                loss=accum_loss,
                lr=lr,
                grad_norm=grad_norm,
                step_ms=1000 * dt / n,
                tokens_per_sec=tokens_per_step * n / dt if dt > 0 else 0.0,
                tokens_seen=(step + 1) * tokens_per_step,
            )
            t_step = time.time()

        is_last = step + 1 == final_step
        if (step + 1) % cfg.eval_interval == 0 or is_last:
            val_loss = evaluate(model, val_sampler, cfg.eval_iters, device_info, amp_enabled)
            logger.metrics("val", step=step + 1, loss=val_loss, ppl=float(torch.exp(torch.tensor(val_loss))))
            t_step = time.time()
        else:
            val_loss = None

        alias = "early" if step + 1 == cfg.early_ckpt_step else None
        if (step + 1) % cfg.ckpt_interval == 0 or is_last or alias:
            ckpts.save(
                step + 1,
                val_loss=val_loss,
                alias=alias,
                model=raw_model,
                optimizer=optimizer,
                scheduler_state=schedule.state_dict(),
                data_state=train_sampler.state_dict(),
                metrics={"train_loss": accum_loss, "val_loss": val_loss},
                **ckpt_common,
            )
            t_step = time.time()

    # An early checkpoint is DPO's source of "rejected" samples; short runs may
    # never reach early_ckpt_step, so guarantee one exists.
    if not ckpts.early_path.exists():
        import shutil

        shutil.copyfile(ckpts.last_path, ckpts.early_path)
    if not ckpts.best_path.exists():
        import shutil

        shutil.copyfile(ckpts.last_path, ckpts.best_path)

    elapsed = time.time() - t_start
    result = StageResult(
        stage="pretrain",
        out_dir=out_dir,
        last_ckpt=ckpts.last_path,
        best_ckpt=ckpts.best_path,
        steps=final_step,
        final_train_loss=last_loss,
        best_val_loss=ckpts.best_val_loss,
        tokens_seen=final_step * tokens_per_step,
        elapsed_sec=elapsed,
    )
    logger.metrics(
        "stage_done",
        step=final_step,
        train_loss=last_loss,
        best_val_loss=ckpts.best_val_loss,
        tokens=result.tokens_seen,
        elapsed_sec=round(elapsed, 2),
    )
    logger.close()
    return result
