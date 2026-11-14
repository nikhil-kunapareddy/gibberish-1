"""The evaluation stage: perplexity + judged samples, written to eval.json.

Input contract:  a checkpoint (any stage) + the prepared data dir it was
                 trained against.
Output contract: `<out_dir>/eval.json` and `<out_dir>/samples.jsonl`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..data.loader import BatchSampler
from ..data.prepare import validate_data_dir, write_jsonl
from ..data.tokenizer import BPETokenizer
from ..inference.generate import generate_text
from ..utils.checkpoint import assert_tokenizer_match, load_model_from_checkpoint
from ..utils.device import resolve_device, use_amp
from ..utils.logging import RunLogger
from ..utils.seed import seed_everything
from .judge import AXES, aggregate, build_judge
from .perplexity import compute_perplexity
from .prompts import get_prompts


def run_eval(cfg, full_config: dict | None = None, quiet: bool = False) -> dict:
    t0 = time.time()
    meta = validate_data_dir(cfg.data_dir)
    seed_everything(cfg.seed)
    device_info = resolve_device(cfg.device)
    amp_enabled = use_amp(device_info, cfg.amp)

    out_dir = Path(cfg.out_dir)
    logger = RunLogger(out_dir, cfg.name, config=full_config)

    model, ckpt = load_model_from_checkpoint(cfg.checkpoint, device_info.device)
    assert_tokenizer_match(ckpt, meta["tokenizer"]["hash"], where=str(cfg.checkpoint))
    tokenizer = BPETokenizer.load(Path(cfg.data_dir))
    logger.info(
        f"[eval] {cfg.checkpoint} (stage={ckpt['stage']} step={ckpt['step']}) on {device_info}"
    )

    # -- perplexity ---------------------------------------------------------- #
    block = min(cfg.block_size, model.cfg.block_size)
    sampler = BatchSampler(cfg.data_dir, "val", block, cfg.batch_size, device_info.device, cfg.seed)
    ppl = compute_perplexity(model, sampler, cfg.max_batches, device_info, amp_enabled)
    logger.metrics("perplexity", loss=ppl["loss"], ppl=ppl["perplexity"], tokens=ppl["tokens"])

    # -- judged samples ------------------------------------------------------ #
    judge = build_judge(cfg.judge, cfg.judge_model)
    kind = "instruct" if ckpt["stage"] in ("sft", "dpo") else "completion"
    prompts = get_prompts(kind, cfg.judge_prompts)
    logger.info(f"[eval] judge={judge.name} on {len(prompts)} {kind} prompts")

    scores = []
    for prompt in prompts:
        generation = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=cfg.judge_max_new_tokens,
            temperature=cfg.judge_temperature,
            top_p=cfg.judge_top_p,
            top_k=0,
            device_info=device_info,
            amp=amp_enabled,
        )[0]
        scores.append(judge.score(prompt, generation.completion))
    judged = aggregate(scores)
    logger.metrics("judge", **{axis: judged[axis] for axis in AXES}, overall=judged["overall"])

    write_jsonl([s.to_dict() for s in scores], out_dir / "samples.jsonl")
    report = {
        "checkpoint": str(cfg.checkpoint),
        "stage": ckpt["stage"],
        "step": ckpt["step"],
        "data_dir": str(cfg.data_dir),
        "tokenizer_hash": meta["tokenizer"]["hash"],
        "model": {k: ckpt["model_config"][k] for k in ("n_layer", "n_head", "d_model", "vocab_size")},
        "non_embedding_params": model.num_params(include_embeddings=False),
        "perplexity": ppl,
        "judge": {"name": judge.name, "prompt_kind": kind, **judged},
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2))
    logger.info(f"[eval] wrote {out_dir / 'eval.json'}")
    logger.close()
    return report
