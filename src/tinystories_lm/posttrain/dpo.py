"""Direct Preference Optimization against a frozen reference model.

**Read this before drawing conclusions from the numbers.** TinyStories ships no
preference data, so the pairs here are synthetic: `chosen` is the story's real
continuation and `rejected` is a high-temperature sample from an early,
barely-trained checkpoint. That makes the preference trivially learnable and
says nothing about whether the model got better. At 1-33M parameters there is
not enough capability for preference optimisation to improve anything. This
stage exists so the code path -- pair construction, frozen reference, the DPO
loss, the logging -- is exercised and tested before it runs on a model where it
matters.

Loss (Rafailov et al., 2023):

    L = -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )

where pi/ref are *summed* completion log-probs under the policy and the frozen
reference. The reference is a snapshot of the SFT model: same weights, frozen,
eval mode, no gradients.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.prepare import read_jsonl, validate_data_dir, write_jsonl
from ..data.tokenizer import BPETokenizer
from ..inference.generate import generate
from ..model.config import ModelConfig
from ..model.transformer import IGNORE_INDEX, Transformer
from ..pretrain.optim import build_optimizer
from ..pretrain.schedule import CosineSchedule
from ..pretrain.train import StageResult
from ..utils.checkpoint import (
    CheckpointManager,
    assert_tokenizer_match,
    load_checkpoint,
    load_model_from_checkpoint,
    restore_training_state,
)
from ..utils.device import autocast_ctx, resolve_device, synchronize, use_amp
from ..utils.logging import RunLogger
from ..utils.seed import seed_everything
from .sft import build_example

# --------------------------------------------------------------------------- #
# Pair synthesis
# --------------------------------------------------------------------------- #


def synthesize_pairs(
    cfg,
    tokenizer: BPETokenizer,
    logger=None,
    device_info=None,
) -> list[dict]:
    """Build `{prompt, chosen, rejected}` rows and write them to `cfg.pairs_file`.

    The prompt is an instruction block plus the first `pair_prompt_tokens`
    tokens of the real story, so it is on-distribution for the SFT model that
    DPO starts from.
    """
    device_info = device_info or resolve_device(cfg.device)
    meta = validate_data_dir(cfg.data_dir, need_sft=True)
    rows = read_jsonl(Path(cfg.data_dir) / meta["sft"]["train_file"])
    if not rows:
        raise RuntimeError(f"no SFT rows in {cfg.data_dir}; cannot synthesize preference pairs")

    sampler_model, ckpt = load_model_from_checkpoint(cfg.sampler_ckpt, device_info.device)
    assert_tokenizer_match(ckpt, meta["tokenizer"]["hash"], where=str(cfg.sampler_ckpt))
    if logger:
        logger.info(
            f"[dpo] sampling 'rejected' continuations from {cfg.sampler_ckpt} "
            f"(step {ckpt['step']}, T={cfg.pair_temperature})"
        )

    pairs: list[dict] = []
    for row in rows:
        if len(pairs) >= cfg.num_pairs:
            break
        prompt_ids = tokenizer.encode(row["prompt"])
        story_ids = tokenizer.encode(row["completion"])
        if len(story_ids) < cfg.pair_prompt_tokens + cfg.pair_max_new_tokens:
            continue
        lead_in, chosen_ids = (
            story_ids[: cfg.pair_prompt_tokens],
            story_ids[cfg.pair_prompt_tokens : cfg.pair_prompt_tokens + cfg.pair_max_new_tokens],
        )
        context = prompt_ids + lead_in
        if len(context) + cfg.pair_max_new_tokens + 1 > sampler_model.cfg.block_size:
            continue
        idx = torch.tensor([context], dtype=torch.long, device=device_info.device)
        out = generate(
            sampler_model,
            idx,
            cfg.pair_max_new_tokens,
            temperature=cfg.pair_temperature,
            top_k=cfg.pair_top_k,
            use_cache=True,
            eot_id=tokenizer.eot_id,
            stop_at_eot=False,
        )
        rejected_ids = out[0, len(context) :].tolist()
        pairs.append(
            {
                "prompt": tokenizer.decode(context),
                "chosen": tokenizer.decode(chosen_ids),
                "rejected": tokenizer.decode(rejected_ids),
                "source_hash": row.get("hash"),
            }
        )

    if not pairs:
        raise RuntimeError(
            "synthesized 0 preference pairs: every story was shorter than "
            f"pair_prompt_tokens + pair_max_new_tokens "
            f"({cfg.pair_prompt_tokens} + {cfg.pair_max_new_tokens})"
        )
    write_jsonl(pairs, cfg.pairs_file)
    if logger:
        logger.info(f"[dpo] wrote {len(pairs)} preference pairs -> {cfg.pairs_file}")
    return pairs


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class PreferenceDataset:
    """Pairs tokenised into two prompt-masked sequences each."""

    def __init__(
        self,
        pairs_file: str | Path,
        tokenizer: BPETokenizer,
        max_seq_len: int,
        device: torch.device | str = "cpu",
        seed: int = 1337,
    ) -> None:
        rows = read_jsonl(pairs_file)
        self.device = torch.device(device)
        self.pad_id = tokenizer.eot_id
        self.items: list[dict] = []
        for row in rows:
            prompt_ids = tokenizer.encode(row["prompt"])
            chosen = build_example(prompt_ids, tokenizer.encode(row["chosen"]), tokenizer.eot_id, max_seq_len)
            rejected = build_example(prompt_ids, tokenizer.encode(row["rejected"]), tokenizer.eot_id, max_seq_len)
            if chosen is None or rejected is None:
                continue
            self.items.append({"chosen": chosen, "rejected": rejected})
        if not self.items:
            raise ValueError(f"{pairs_file} produced no usable preference pairs")
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.items)

    def collate(self, indices: list[int]) -> dict[str, torch.Tensor]:
        """Chosen and rejected are padded to a common width and stacked.

        Both halves share one padded width so a single forward pass covers the
        batch; IGNORE_INDEX keeps padding out of the log-probs.
        """
        rows = [self.items[i] for i in indices]
        width = max(len(r[k][0]) for r in rows for k in ("chosen", "rejected"))
        out = {}
        for key in ("chosen", "rejected"):
            x = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
            y = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
            for r, row in enumerate(rows):
                xi, yi = row[key]
                x[r, : len(xi)] = torch.tensor(xi, dtype=torch.long)
                y[r, : len(yi)] = torch.tensor(yi, dtype=torch.long)
            out[f"{key}_x"] = x.to(self.device)
            out[f"{key}_y"] = y.to(self.device)
        return out

    def batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, len(self), (batch_size,), generator=self.generator).tolist()
        return self.collate(idx)

    def eval_batches(self, n_batches: int, batch_size: int):
        for b in range(n_batches):
            lo = (b * batch_size) % len(self)
            yield self.collate([(lo + i) % len(self) for i in range(batch_size)])

    def state_dict(self) -> dict:
        return {"generator": self.generator.get_state(), "n_pairs": len(self)}

    def load_state_dict(self, state: dict) -> None:
        g = state["generator"]
        self.generator.set_state(g if isinstance(g, torch.Tensor) else torch.tensor(g, dtype=torch.uint8))


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def sequence_logprobs(model: Transformer, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Summed log p(y_t | x_<=t) over supervised positions. Returns (B,)."""
    logits, _ = model(x)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    mask = y != IGNORE_INDEX
    safe_y = y.masked_fill(~mask, 0)
    token_logprobs = logprobs.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The DPO objective plus the diagnostics worth logging."""
    pi_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = pi_logratio - ref_logratio
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
    chosen_reward = beta * (policy_chosen - ref_chosen).detach()
    rejected_reward = beta * (policy_rejected - ref_rejected).detach()
    metrics = {
        "reward_chosen": chosen_reward.mean(),
        "reward_rejected": rejected_reward.mean(),
        "reward_margin": (chosen_reward - rejected_reward).mean(),
        "reward_accuracy": (chosen_reward > rejected_reward).float().mean(),
        "logits": logits.mean().detach(),
    }
    return losses.mean(), metrics


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate_dpo(policy, reference, dataset, iters, batch_size, beta, device_info, amp) -> dict:
    was_training = policy.training
    policy.eval()
    totals: dict[str, float] = {}
    n = 0
    for batch in dataset.eval_batches(iters, batch_size):
        with autocast_ctx(device_info, amp):
            loss, metrics = _pair_step(policy, reference, batch, beta, 0.0)
        totals["loss"] = totals.get("loss", 0.0) + loss.item()
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1
    policy.train(was_training)
    return {k: v / max(1, n) for k, v in totals.items()}


def _pair_step(policy, reference, batch, beta: float, label_smoothing: float):
    policy_chosen = sequence_logprobs(policy, batch["chosen_x"], batch["chosen_y"])
    policy_rejected = sequence_logprobs(policy, batch["rejected_x"], batch["rejected_y"])
    with torch.no_grad():
        ref_chosen = sequence_logprobs(reference, batch["chosen_x"], batch["chosen_y"])
        ref_rejected = sequence_logprobs(reference, batch["rejected_x"], batch["rejected_y"])
    return dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, label_smoothing)


def freeze_reference(model: Transformer) -> Transformer:
    """A frozen, eval-mode copy of the policy at initialisation."""
    reference = copy.deepcopy(model)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    return reference


def run_dpo(cfg, full_config: dict | None = None, quiet: bool = False) -> StageResult:
    t_start = time.time()
    meta = validate_data_dir(cfg.data_dir, need_sft=True)
    seed_everything(cfg.seed)
    device_info = resolve_device(cfg.device)
    amp_enabled = use_amp(device_info, cfg.amp)

    out_dir = Path(cfg.out_dir)
    logger = RunLogger(out_dir, cfg.name, cfg.wandb, cfg.wandb_project, full_config)

    base = load_checkpoint(cfg.init_from, map_location="cpu")
    assert_tokenizer_match(base, meta["tokenizer"]["hash"], where=str(cfg.init_from))
    model_cfg = ModelConfig.from_dict(base["model_config"])
    policy = Transformer(model_cfg)
    policy.load_state_dict(base["model_state"])
    policy.to(device_info.device)
    reference = freeze_reference(policy)
    logger.info(
        f"[dpo] policy+reference from {cfg.init_from} "
        f"(stage={base['stage']} step={base['step']}) on {device_info}"
    )

    tokenizer = BPETokenizer.load(Path(cfg.data_dir))
    if not Path(cfg.pairs_file).exists():
        # Pairs are regenerated whenever the file is missing. They live under
        # the run directory because they depend on `sampler_ckpt` -- delete the
        # file (or the run) to rebuild them against a different checkpoint.
        synthesize_pairs(cfg, tokenizer, logger, device_info)
    dataset = PreferenceDataset(
        cfg.pairs_file, tokenizer, cfg.max_seq_len, device_info.device, cfg.seed
    )
    logger.info(f"[dpo] {len(dataset)} preference pairs from {cfg.pairs_file}")

    optimizer = build_optimizer(
        policy, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps, device_type=device_info.type,
    )
    schedule = CosineSchedule(cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
    ckpts = CheckpointManager(out_dir, cfg.keep_last_k, stage="dpo")

    start_step = 0
    resume_path = ckpts.resolve_resume(cfg.resume)
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location=device_info.device)
        start_step = restore_training_state(ckpt, model=policy, optimizer=optimizer)
        if ckpt.get("data_state"):
            dataset.load_state_dict(ckpt["data_state"])
        ckpts.best_val_loss = ckpt.get("best_val_loss") or float("inf")
        logger.info(f"[dpo] resumed from {resume_path} at step {start_step}")

    raw_policy = policy
    if cfg.compile:
        policy = torch.compile(policy)

    ckpt_common = dict(
        model_config=model_cfg,
        tokenizer=meta["tokenizer"],
        data_meta={"data_dir": str(cfg.data_dir), "pairs_file": str(cfg.pairs_file),
                   "init_from": str(cfg.init_from), "sampler_ckpt": str(cfg.sampler_ckpt)},
        config=full_config or {},
    )

    policy.train()
    last_loss = float("nan")
    t_step = time.time()
    for step in range(start_step, cfg.max_steps):
        lr = schedule.apply(optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        step_metrics: dict[str, float] = {}
        for _ in range(cfg.grad_accum_steps):
            batch = dataset.batch(cfg.batch_size)
            with autocast_ctx(device_info, amp_enabled):
                loss, metrics = _pair_step(policy, reference, batch, cfg.beta, cfg.label_smoothing)
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item() / cfg.grad_accum_steps
            for k, v in metrics.items():
                step_metrics[k] = step_metrics.get(k, 0.0) + float(v) / cfg.grad_accum_steps

        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip))
        optimizer.step()
        last_loss = accum_loss

        if (step + 1) % cfg.log_interval == 0 or step == start_step:
            synchronize(device_info)
            dt = time.time() - t_step
            logger.metrics(
                "train", step=step + 1, loss=accum_loss, lr=lr, grad_norm=grad_norm,
                step_ms=1000 * dt / max(1, cfg.log_interval), **step_metrics,
            )
            t_step = time.time()

        is_last = step + 1 == cfg.max_steps
        val_loss = None
        if (step + 1) % cfg.eval_interval == 0 or is_last:
            stats = evaluate_dpo(policy, reference, dataset, cfg.eval_iters, cfg.batch_size,
                                 cfg.beta, device_info, amp_enabled)
            val_loss = stats["loss"]
            logger.metrics("val", step=step + 1, **stats)
            t_step = time.time()

        if (step + 1) % cfg.ckpt_interval == 0 or is_last:
            ckpts.save(
                step + 1, val_loss=val_loss, model=raw_policy, optimizer=optimizer,
                scheduler_state=schedule.state_dict(), data_state=dataset.state_dict(),
                metrics={"train_loss": accum_loss, "val_loss": val_loss, **step_metrics},
                **ckpt_common,
            )
            t_step = time.time()

    if not ckpts.best_path.exists():
        import shutil

        shutil.copyfile(ckpts.last_path, ckpts.best_path)

    elapsed = time.time() - t_start
    result = StageResult(
        stage="dpo", out_dir=out_dir, last_ckpt=ckpts.last_path, best_ckpt=ckpts.best_path,
        steps=cfg.max_steps, final_train_loss=last_loss, best_val_loss=ckpts.best_val_loss,
        tokens_seen=0, elapsed_sec=elapsed,
    )
    logger.metrics("stage_done", step=cfg.max_steps, train_loss=last_loss,
                   elapsed_sec=round(elapsed, 2))
    logger.close()
    return result
