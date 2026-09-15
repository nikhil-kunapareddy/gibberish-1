"""Autoregressive generation, with and without a KV cache.

Both paths must produce identical tokens for the same prompt and sampling
settings -- that is gate 8, and it is the only way to know the cache is not
quietly misaligning positions. The cache is a speed optimisation and nothing
else, so when the context outgrows `block_size` both paths do the same thing:
slide the window and, for the cached path, rebuild the cache from the window.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ..data.tokenizer import BPETokenizer
from ..model.transformer import Transformer
from ..utils.device import DeviceInfo, autocast_ctx, device_info_for, synchronize
from .sampling import sample_from_logits


@torch.no_grad()
def generate(
    model: Transformer,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    use_cache: bool = True,
    eot_id: int | None = None,
    stop_at_eot: bool = False,
    generator: torch.Generator | None = None,
    device_info: DeviceInfo | None = None,
    amp: bool = False,
) -> torch.Tensor:
    """idx: (B, T) prompt tokens -> (B, T + n) with the continuation appended.

    Every row of `idx` must be the same length; batch generation here means
    "same prompt, several samples", which is all the eval harness needs.

    Both paths keep the *same* context window, which is what makes gate 8 hold
    at any length: when the context fills `block_size`, both drop to the most
    recent `block_size // 2` tokens and continue, and the cache is rebuilt from
    that window. Halving rather than shaving one token keeps the rebuild
    amortised -- it happens once every `block_size // 2` tokens instead of on
    every step.
    """
    was_training = model.training
    model.eval()
    block = model.cfg.block_size
    keep = max(1, block // 2)

    output = idx.clone()  # full history, returned to the caller
    context = idx[:, -(block - 1) :] if idx.size(1) >= block else idx  # what the model sees
    cache = model.new_kv_cache(idx.size(0), block) if use_cache else None
    ctx = autocast_ctx(device_info, amp) if device_info is not None else nullcontext()
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

    with ctx:
        for _ in range(max_new_tokens):
            if context.size(1) >= block:  # no room for the next token
                context = context[:, -keep:]
                if cache is not None:
                    cache.reset()

            if cache is None or cache.pos == 0:
                model_input = context  # full pass: no cache, or a (re)prefill
            else:
                model_input = context[:, -1:]  # decode: only the new token
            logits, _ = model(model_input, cache=cache, last_token_only=True)

            next_token = sample_from_logits(
                logits[:, -1, :], temperature, top_k, top_p, greedy, generator
            )
            if eot_id is not None and stop_at_eot:
                # A finished row keeps emitting EOT so the batch stays aligned.
                next_token = torch.where(
                    finished.unsqueeze(1), torch.full_like(next_token, eot_id), next_token
                )
                finished |= next_token.squeeze(1) == eot_id

            output = torch.cat([output, next_token], dim=1)
            context = torch.cat([context, next_token], dim=1)
            if stop_at_eot and bool(finished.all()):
                break

    model.train(was_training)
    return output


# --------------------------------------------------------------------------- #
# Text-level API
# --------------------------------------------------------------------------- #


@dataclass
class Generation:
    text: str
    prompt: str
    completion: str
    n_new_tokens: int
    seconds: float

    @property
    def tokens_per_sec(self) -> float:
        return self.n_new_tokens / self.seconds if self.seconds > 0 else float("nan")


def generate_text(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    greedy: bool = False,
    num_samples: int = 1,
    use_cache: bool = True,
    stop_at_eot: bool = True,
    device_info: DeviceInfo | None = None,
    amp: bool = False,
) -> list[Generation]:
    """Importable API: prompt in, decoded continuations out."""
    device_info = device_info or device_info_for(model)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.eot_id]  # unconditional sample
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device_info.device)
    idx = idx.unsqueeze(0).expand(num_samples, -1).contiguous()

    synchronize(device_info)
    t0 = time.time()
    out = generate(
        model,
        idx,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        use_cache=use_cache,
        eot_id=tokenizer.eot_id,
        stop_at_eot=stop_at_eot,
        device_info=device_info,
        amp=amp,
    )
    synchronize(device_info)
    elapsed = time.time() - t0

    results = []
    n_prompt = idx.size(1)
    for row in out:
        completion_ids = row[n_prompt:].tolist()
        if stop_at_eot and tokenizer.eot_id in completion_ids:
            completion_ids = completion_ids[: completion_ids.index(tokenizer.eot_id)]
        completion = tokenizer.decode(completion_ids)
        results.append(
            Generation(
                text=prompt + completion,
                prompt=prompt,
                completion=completion,
                n_new_tokens=len(completion_ids),
                seconds=elapsed / max(1, num_samples),
            )
        )
    return results


def profile_generation(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    device_info: DeviceInfo | None = None,
    amp: bool = False,
) -> dict:
    """Time greedy decoding with and without the cache (the `--profile` flag).

    Greedy so the two runs decode the same tokens and the comparison is fair.
    """
    device_info = device_info or device_info_for(model)
    stats = {}
    for use_cache in (False, True):
        # One short warmup pass: the first call pays for lazy kernel init.
        generate_text(model, tokenizer, prompt, 4, greedy=True, use_cache=use_cache,
                      stop_at_eot=False, device_info=device_info, amp=amp)
        synchronize(device_info)
        t0 = time.time()
        out = generate_text(
            model, tokenizer, prompt, max_new_tokens, greedy=True, use_cache=use_cache,
            stop_at_eot=False, device_info=device_info, amp=amp,
        )
        synchronize(device_info)
        elapsed = time.time() - t0
        key = "with_cache" if use_cache else "no_cache"
        stats[key] = {
            "seconds": round(elapsed, 4),
            "tokens": out[0].n_new_tokens,
            "tokens_per_sec": round(out[0].n_new_tokens / elapsed, 1) if elapsed else None,
        }
        stats[f"{key}_text"] = out[0].completion
    no_cache, cached = stats["no_cache"]["seconds"], stats["with_cache"]["seconds"]
    stats["speedup"] = round(no_cache / cached, 2) if cached else None
    stats["identical_output"] = stats["no_cache_text"] == stats["with_cache_text"]
    return stats
