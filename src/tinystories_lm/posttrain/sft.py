"""Supervised fine-tuning on TinyStories-Instruct.

The one thing that has to be right here is the loss mask. An instruction
example is `prompt + completion`, and the model must be scored only on the
completion: training on the prompt teaches it to generate instructions, which
looks exactly like normal training right up until you read a sample. Gate 9
asserts that prompt positions contribute zero gradient.

Masking convention (see `build_example`): with `x = seq[:-1]`, `y = seq[1:]`,
the target at position i is `seq[i+1]`, so the first `len(prompt) - 1` targets
are prompt tokens and get IGNORE_INDEX. Position `len(prompt) - 1` is the last
prompt token *predicting the first completion token*, which is supervised.

Input contract:  a pretrain checkpoint + a prepared data dir with sft_*.jsonl.
Output contract: `<out_dir>/ckpt_{last,best}.pt`, `metrics.jsonl`, `config.yaml`.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..data.prepare import read_jsonl, validate_data_dir
from ..data.tokenizer import BPETokenizer
from ..model.config import ModelConfig
from ..model.transformer import IGNORE_INDEX, Transformer
from ..pretrain.optim import build_optimizer
from ..pretrain.schedule import CosineSchedule
from ..pretrain.train import StageResult
from ..utils.checkpoint import (
    CheckpointManager,
    assert_tokenizer_match,
    load_checkpoint,
    restore_training_state,
)
from ..utils.device import autocast_ctx, resolve_device, synchronize, use_amp
from ..utils.logging import RunLogger
from ..utils.seed import seed_everything


def build_example(
    prompt_ids: list[int], completion_ids: list[int], eot_id: int, max_seq_len: int
) -> tuple[list[int], list[int]] | None:
    """One (input, target) pair with the prompt masked out.

    Returns None when truncation leaves nothing to supervise.
    """
    seq = list(prompt_ids) + list(completion_ids) + [eot_id]
    seq = seq[: max_seq_len + 1]
    if len(seq) < 2 or len(prompt_ids) >= len(seq):
        return None
    x, y = seq[:-1], seq[1:]
    n_masked = len(prompt_ids) - 1
    targets = [IGNORE_INDEX] * n_masked + y[n_masked:]
    if all(t == IGNORE_INDEX for t in targets):
        return None
    return x, targets


class SFTDataset:
    """Pre-tokenised instruction examples, padded per batch."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: BPETokenizer,
        max_seq_len: int,
        device: torch.device | str = "cpu",
        seed: int = 1337,
    ) -> None:
        rows = read_jsonl(jsonl_path)
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.eot_id
        self.examples: list[tuple[list[int], list[int]]] = []
        self.prompt_lens: list[int] = []
        skipped = 0
        prompts = tokenizer.encode_batch([r["prompt"] for r in rows])
        completions = tokenizer.encode_batch([r["completion"] for r in rows])
        for p_ids, c_ids in zip(prompts, completions):
            example = build_example(p_ids, c_ids, tokenizer.eot_id, max_seq_len)
            if example is None:
                skipped += 1
                continue
            self.examples.append(example)
            self.prompt_lens.append(len(p_ids))
        self.skipped = skipped
        if not self.examples:
            raise ValueError(
                f"{jsonl_path} produced no usable SFT examples at max_seq_len={max_seq_len}"
            )
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a group of examples to the longest in the group.

        Padded positions get IGNORE_INDEX targets, so padding never contributes
        to the loss either.
        """
        rows = [self.examples[i] for i in indices]
        width = max(len(x) for x, _ in rows)
        x = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        y = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
        for r, (xi, yi) in enumerate(rows):
            x[r, : len(xi)] = torch.tensor(xi, dtype=torch.long)
            y[r, : len(yi)] = torch.tensor(yi, dtype=torch.long)
        return x.to(self.device), y.to(self.device)

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self), (batch_size,), generator=self.generator).tolist()
        return self.collate(idx)

    def eval_batches(self, n_batches: int, batch_size: int):
        """Deterministic, non-overlapping batches from the head of the set."""
        for b in range(n_batches):
            lo = (b * batch_size) % len(self)
            idx = [(lo + i) % len(self) for i in range(batch_size)]
            yield self.collate(idx)

    def state_dict(self) -> dict:
        return {"generator": self.generator.get_state(), "n_examples": len(self)}

    def load_state_dict(self, state: dict) -> None:
        g = state["generator"]
        self.generator.set_state(g if isinstance(g, torch.Tensor) else torch.tensor(g, dtype=torch.uint8))


@torch.no_grad()
def evaluate_sft(model, dataset: SFTDataset, iters: int, batch_size: int, device_info, amp) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for x, y in dataset.eval_batches(iters, batch_size):
        with autocast_ctx(device_info, amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training)
    return float(sum(losses) / max(1, len(losses)))


def run_sft(cfg, full_config: dict | None = None, quiet: bool = False) -> StageResult:
    t_start = time.time()
    meta = validate_data_dir(cfg.data_dir, need_sft=True)
    seed_everything(cfg.seed)
    device_info = resolve_device(cfg.device)
    amp_enabled = use_amp(device_info, cfg.amp)

    out_dir = Path(cfg.out_dir)
    logger = RunLogger(out_dir, cfg.name, cfg.wandb, cfg.wandb_project, full_config)

    # -- input contract: a pretrain checkpoint, matching this tokenizer ------ #
    base = load_checkpoint(cfg.init_from, map_location="cpu")
    assert_tokenizer_match(base, meta["tokenizer"]["hash"], where=str(cfg.init_from))
    model_cfg = ModelConfig.from_dict(base["model_config"])
    model = Transformer(model_cfg)
    model.load_state_dict(base["model_state"])
    model.to(device_info.device)
    logger.info(
        f"[sft] init from {cfg.init_from} (stage={base['stage']} step={base['step']}) "
        f"on {device_info}"
    )

    tokenizer = BPETokenizer.load(Path(cfg.data_dir))
    train_set = SFTDataset(
        Path(cfg.data_dir) / meta["sft"]["train_file"], tokenizer, cfg.max_seq_len,
        device_info.device, cfg.seed
    )
    val_set = SFTDataset(
        Path(cfg.data_dir) / meta["sft"]["val_file"], tokenizer, cfg.max_seq_len,
        device_info.device, cfg.seed + 1
    )
    logger.info(
        f"[sft] {len(train_set)} train / {len(val_set)} val examples "
        f"(skipped {train_set.skipped} that did not fit in {cfg.max_seq_len} tokens)"
    )

    optimizer = build_optimizer(
        model, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps, device_type=device_info.type,
    )
    schedule = CosineSchedule(cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
    ckpts = CheckpointManager(out_dir, cfg.keep_last_k, stage="sft")

    start_step = 0
    resume_path = ckpts.resolve_resume(cfg.resume)
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location=device_info.device)
        start_step = restore_training_state(ckpt, model=model, optimizer=optimizer)
        if ckpt.get("data_state"):
            train_set.load_state_dict(ckpt["data_state"])
        ckpts.best_val_loss = ckpt.get("best_val_loss") or float("inf")
        logger.info(f"[sft] resumed from {resume_path} at step {start_step}")

    raw_model = model
    if cfg.compile:
        model = torch.compile(model)

    ckpt_common = dict(
        model_config=model_cfg,
        tokenizer=meta["tokenizer"],
        data_meta={"data_dir": str(cfg.data_dir), "sft": meta["sft"], "init_from": str(cfg.init_from)},
        config=full_config or {},
    )

    model.train()
    last_loss = float("nan")
    tokens_seen = 0
    t_step = time.time()
    for step in range(start_step, cfg.max_steps):
        lr = schedule.apply(optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_set.batch(cfg.batch_size)
            with autocast_ctx(device_info, amp_enabled):
                _, loss = model(x, y)
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item() / cfg.grad_accum_steps
            tokens_seen += int((y != IGNORE_INDEX).sum())

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()
        last_loss = accum_loss

        if (step + 1) % cfg.log_interval == 0 or step == start_step:
            synchronize(device_info)
            dt = time.time() - t_step
            logger.metrics(
                "train", step=step + 1, loss=accum_loss, lr=lr, grad_norm=grad_norm,
                supervised_tokens=tokens_seen, step_ms=1000 * dt / max(1, cfg.log_interval),
            )
            t_step = time.time()

        is_last = step + 1 == cfg.max_steps
        val_loss = None
        if (step + 1) % cfg.eval_interval == 0 or is_last:
            val_loss = evaluate_sft(model, val_set, cfg.eval_iters, cfg.batch_size, device_info, amp_enabled)
            logger.metrics("val", step=step + 1, loss=val_loss)
            t_step = time.time()

        if (step + 1) % cfg.ckpt_interval == 0 or is_last:
            ckpts.save(
                step + 1, val_loss=val_loss, model=raw_model, optimizer=optimizer,
                scheduler_state=schedule.state_dict(), data_state=train_set.state_dict(),
                metrics={"train_loss": accum_loss, "val_loss": val_loss}, **ckpt_common,
            )
            t_step = time.time()

    if not ckpts.best_path.exists():
        import shutil

        shutil.copyfile(ckpts.last_path, ckpts.best_path)

    elapsed = time.time() - t_start
    result = StageResult(
        stage="sft", out_dir=out_dir, last_ckpt=ckpts.last_path, best_ckpt=ckpts.best_path,
        steps=cfg.max_steps, final_train_loss=last_loss, best_val_loss=ckpts.best_val_loss,
        tokens_seen=tokens_seen, elapsed_sec=elapsed,
    )
    logger.metrics(
        "stage_done", step=cfg.max_steps, train_loss=last_loss,
        best_val_loss=ckpts.best_val_loss, supervised_tokens=tokens_seen,
        elapsed_sec=round(elapsed, 2),
    )
    logger.close()
    return result
