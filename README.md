# tinystories-lm

An end-to-end LLM pipeline built from scratch on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) corpus:
data preparation → pretraining → post-training (SFT + DPO) → evaluation →
inference. Every stage is config-driven, independently runnable, and
independently tested.

```
make smoke            # the whole pipeline + nine correctness gates
make smoke DEVICE=cpu # ... forced onto CPU
```

Measured on an M-series laptop from a clean checkout: **57s on CPU**, **1.7 min
on MPS**, against a budget of 30 and 10 minutes respectively. (MPS is slower
here only because half the time is DPO pair synthesis — 64 sequential 64-token
generations, where a 1.5M-parameter model is latency-bound and per-op overhead
dominates. Every other stage is faster on MPS.)

**This milestone is a smoke test, not a quality run.** The goal is to prove that
every stage runs, hands off correctly to the next, and is verifiably correct.
The model it produces writes near-gibberish. That is a pass — the
[correctness gates](#correctness-gates) are the success criterion.

---

## Table of contents

- [Quick start](#quick-start)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [Configuration](#configuration)
- [Correctness gates](#correctness-gates)
- [Checkpoints](#checkpoints)
- [Deviations from the TinyStories paper](#deviations-from-the-tinystories-paper)
- [What DPO is doing here](#what-dpo-is-doing-here)
- [Evaluation and the judge](#evaluation-and-the-judge)
- [Scaling up](#scaling-up)
- [Repository layout](#repository-layout)
- [Dependencies](#dependencies)

---

## Quick start

```bash
pip install -e ".[dev]"        # or: pip install torch tokenizers numpy pyyaml tqdm pytest
make smoke                     # everything, end to end, with the gates
```

`make smoke` needs no API key and no GPU. It downloads ~45MB of TinyStories the
first time and caches it in `data/raw/`.

Stages can also be run one at a time:

```bash
make data           # 1. download + tokenize + split      -> data/smoke/
make pretrain       # 2. pretrain the 1M model            -> runs/pretrain-smoke/
make sft            # 3. instruction tuning               -> runs/sft-smoke/
make dpo            # 4. DPO (mechanics only, see below)  -> runs/dpo-smoke/
make evaluate       # 5. perplexity + judge               -> runs/eval-smoke/
make generate PROMPT="Once upon a time"
make profile        # tokens/sec with and without the KV cache
make test           # the full test suite
```

Every target is a thin wrapper over `scripts/`, and every script is a thin
wrapper over `src/`. There is no logic in `scripts/` or in the `Makefile`.

---

## The pipeline, stage by stage

Stages hand off **through artifacts on disk**, never in-memory objects. Each
stage validates its input contract on entry and fails with an actionable
message if the previous stage did not run.

### 1. Data — `scripts/prepare_data.py`

| | |
|---|---|
| **In** | `configs/data/*.yaml` |
| **Out** | `data/<name>/{train,val}.bin`, `tokenizer.json`, `sft_{train,val}.jsonl`, `meta.json` |

- Downloads `roneneldan/TinyStories` and `roneneldan/TinyStories-Instruct`
  (CDLA-Sharing-1.0) over plain HTTPS and caches them in `data/raw/`. Never
  re-downloads. `max_download_bytes` issues an HTTP range request so the smoke
  run pulls 24MB instead of the 1.9GB training file.
- **Trains a byte-level BPE at 8192 tokens** rather than reusing GPT-2's 50257.
  This matters: at `d_model=512`, a 50k vocab puts ~25.7M parameters in the
  embedding alone, which would dominate a 33M model. At 8192 the embedding is
  4.2M. `vocab_size` is a config value and the tokenizer is swappable — nothing
  downstream hardcodes it, everything reads it from `meta.json` or the checkpoint.
- The tokenizer is trained on the **train split only**. Fitting BPE merges on
  validation text is a small but real leak.
- **Splits by story, not by token.** Each story's split is a function of the hash
  of its text (salted by `split_seed`), so the assignment is reproducible,
  order-independent, and auditable — the sorted hashes are written to
  `meta.json`. Tokens are concatenated *after* the split, so no story can
  straddle the boundary. Duplicate stories collapse to one copy and therefore
  cannot appear on both sides.
- Output is a flat `uint16` token stream, one `<|endoftext|>` after each story,
  memory-mapped at load time. No padding and no pre-built windows: the sampler
  picks the context length, so changing `block_size` is a config change, not a
  re-preparation.

### 2. Pretraining — `scripts/pretrain.py`

| | |
|---|---|
| **In** | a prepared data dir + `configs/model/*.yaml` + `configs/pretrain/*.yaml` |
| **Out** | `runs/<name>/ckpt_{last,best,early}.pt`, `metrics.jsonl`, `config.yaml` |

A hand-written loop — no Trainer, no Lightning, no Accelerate. AdamW with
weight decay on projection matrices only (not norms, biases, or embeddings),
linear warmup into cosine decay, gradient clipping at 1.0, gradient
accumulation, bf16 autocast where the device supports it, and `torch.compile`
behind an off-by-default flag (slow to warm up, flaky on MPS).

Logs train loss, val loss, LR, grad norm, tokens/sec and step time to stdout and
to `metrics.jsonl`. Checkpoints every N steps, keeping the last K plus the
best-by-val-loss. W&B is optional and off by default (`--pretrain.wandb=true`).

### 3. SFT — `scripts/sft.py`

| | |
|---|---|
| **In** | a pretrain checkpoint + `sft_{train,val}.jsonl` |
| **Out** | `runs/sft-*/ckpt_{last,best}.pt` |

Instruction tuning on TinyStories-Instruct, where each record is an instruction
block (`Features:` / `Words:` / `Summary:` / `Random sentence:`) followed by
`Story:` and the story. The prompt is everything up to and including the
`Story:` marker; the completion is the story.

**The loss is masked to the completion.** Training on the prompt teaches the
model to generate instructions, which looks exactly like normal training until
you read a sample — so gate 9 asserts mechanically that prompt positions
contribute nothing. Padding is masked the same way.

### 4. DPO — `scripts/dpo.py`

| | |
|---|---|
| **In** | the SFT checkpoint (policy *and* frozen reference) + an early pretrain checkpoint (sampler) |
| **Out** | `runs/dpo-*/dpo_pairs.jsonl`, `runs/dpo-*/ckpt_last.pt` |

See [What DPO is doing here](#what-dpo-is-doing-here) before reading anything
into the numbers.

### 5. Evaluation — `scripts/evaluate.py`

| | |
|---|---|
| **In** | any checkpoint + the data dir it was trained against |
| **Out** | `runs/eval-*/eval.json`, `samples.jsonl` |


Validation perplexity over non-overlapping windows (a property of the corpus,
not of a sampling seed), plus an LLM-as-judge scoring completions separately on
**grammar, creativity and consistency** — the paper's decomposition, kept
because it is what makes small-model output legible. See
[Evaluation and the judge](#evaluation-and-the-judge).

### 6. Inference — `scripts/generate.py`

Autoregressive generation with a KV cache; temperature, top-k, top-p, greedy.
`--profile` reports tokens/sec with and without the cache, decoding greedily in
both so the comparison is like for like — and asserts that the two produce the
same text:

```bash
$ make profile CKPT=runs/pretrain-smoke/ckpt_best.pt
{
  "no_cache":   {"seconds": 8.83, "tokens": 128, "tokens_per_sec": 14.5},
  "with_cache": {"seconds": 1.19, "tokens": 128, "tokens_per_sec": 107.9},
  "speedup": 7.45,
  "identical_output": true
}
```

Also importable:

```python
from tinystories_lm.utils.checkpoint import load_model_from_checkpoint
from tinystories_lm.data.tokenizer import BPETokenizer
from tinystories_lm.inference.generate import generate_text

model, ckpt = load_model_from_checkpoint("runs/sft-smoke/ckpt_best.pt")
tok = BPETokenizer.load("data/smoke")
print(generate_text(model, tok, "Once upon a time", max_new_tokens=100)[0].text)
```

---

## Configuration

A run is defined by its config files. Every knob lives in a typed dataclass
(`src/tinystories_lm/utils/config.py`, plus `model/config.py`), is loaded from
YAML, and is overridable from the command line as `--<section>.<field>=<value>`:

```bash
python scripts/pretrain.py --pretrain configs/pretrain/smoke.yaml \
    --model configs/model/tiny-10m.yaml \
    --pretrain.lr=3e-4 --pretrain.max_steps=2000 --model.n_layer=8
```

Sections are `model`, `data`, `pretrain`, `sft`, `dpo`, `eval`, `generate`.
Because sections are typed:

- a typo in a YAML key is an error at load time, not a silently ignored setting;
- an override naming a section the command did not load is an error
  (`--model.n_layer=8` passed to `scripts/sft.py` fails loudly rather than
  doing nothing);
- values are coerced to the declared type, so `--pretrain.lr=3e-4` is a float
  and `--pretrain.amp=false` stays a bool.

Each stage writes the fully resolved config to `<out_dir>/config.yaml`, so
every run records exactly what produced it.

---

## Correctness gates

Nine gates, each a real test in `tests/`. `make smoke` runs them and fails
loudly; `scripts/smoke_test.py` prints a per-gate pass/fail table.

| # | Gate | Why it exists | Tests |
|---|------|---------------|-------|
| 1 | **Tokenizer roundtrip** — encode→decode 1000 random stories, exact match | A lossy tokenizer caps quality in a way no loss curve reveals | `test_data.py::test_gate1_*` |
| 2 | **Off-by-one** — `y[i,t] == x[i,t+1]` across the whole batch | The single most common pipeline bug | `test_data.py::test_gate2_*` |
| 3 | **No split leakage** — train ∩ val story hashes is empty | Validation loss is meaningless without it | `test_data.py::test_gate3_*` |
| 4 | **Untrained loss** — within 5% of `ln(vocab_size)` | Catches bad init and label misalignment | `test_model.py::test_gate4_*` |
| 5 | **Single-batch overfit** — one fixed batch, 200 steps, final loss < 0.1 | Catches most silent model bugs; the first thing that should work | `test_train.py::test_gate5_*` |
| 6 | **Causal mask integrity** — perturb token `t`, logits at every position `< t` are unchanged | A leaky mask produces suspiciously good loss and is otherwise invisible | `test_model.py::test_gate6_*` |
| 7 | **Resume equivalence** — 20 + 20 resumed steps == 40 straight steps | Proves the checkpoint captures the *entire* run state | `test_train.py::test_gate7_*` |
| 8 | **KV cache equivalence** — 50 greedy tokens, cached == uncached | The only way to know the cache is not misaligning positions | `test_inference.py::test_gate8_*` |
| 9 | **SFT mask correctness** — prompt positions contribute zero loss | A silent bug that looks like normal training | `test_posttrain.py::test_gate9_*` |

Gate 7 is checked on both the loss trace *and* the weights, bit for bit
(`atol=0, rtol=0`). Gate 5's fixture is `fit_single_batch()` in
`pretrain/train.py` — it lives in the training module, not the test file,
because it is the first thing to reach for when a model change makes a loss
curve look wrong.

```bash
make test        # everything, including the gates that train for a few hundred steps
make test-fast   # skips those
```

---

## Checkpoints

Every checkpoint is self-describing: `load_model_from_checkpoint(path)` gives
back a working model with no other inputs. Each one carries

- model weights **and** the full `ModelConfig` that shaped them,
- optimizer state, scheduler state, step count, best-val-loss, and the stage
  that wrote it,
- RNG state for python / numpy / torch / accelerator **and** the data sampler's
  own generator — which is what makes gate 7 exact,
- the tokenizer hash and data metadata, so a checkpoint meeting a tokenizer it
  was not trained with is an error rather than quiet gibberish.

```bash
python scripts/evaluate.py -v --eval.checkpoint=runs/sft-smoke/ckpt_best.pt
```

---

## Deviations from the TinyStories paper

The paper trains GPT-Neo architectures. This repo uses modern components
throughout, so that scaling the same code up is clean:

| Component | Paper (GPT-Neo) | Here | Why |
|---|---|---|---|
| Normalisation | LayerNorm, post-norm | **RMSNorm, pre-norm** | Pre-norm trains stably at depth without warmup tricks; RMSNorm is cheaper and loses nothing |
| Positions | Learned absolute | **RoPE** | Relative by construction, and no position table to resize when `block_size` changes |
| MLP | GELU, 4× width | **SwiGLU, 8/3× width** | Better quality per parameter; the width keeps the parameter count level |
| Biases | Present | **None** | No measurable quality cost at this scale |
| Embeddings | Untied | **Tied** | At an 8k vocab the embedding is a large fraction of a small model |
| Tokenizer | GPT-Neo's 50257 | **Custom 8192 BPE** | See [Data](#1-data--scriptsprepare_datapy) |
| Attention | Some local/windowed layers | **Full causal attention everywhere** | Attention variants are explicitly out of scope for this milestone |

`configs/model/tiny-33m.yaml` matches the paper's largest model on
**non-embedding parameter count** (32.1M) rather than on architecture. Every
model prints its breakdown at init:

```
Transformer[tiny-33m]: n_layer=10 n_head=8 d_model=512 hidden=1408 vocab=8192 block=512 tied=True
  params total           36,317,696  (36.32M)
  params non-embedding   32,123,392  (32.12M)
    embedding             4,194,304
    attention            10,485,760
    mlp                  21,626,880
    norm                     10,752
```

---

## What DPO is doing here

**DPO in this repo validates mechanics only. It does not make the model
better, and it is not intended to.**

TinyStories ships no preference data, so the pairs are synthesised:

- `chosen` — the story's real continuation,
- `rejected` — a high-temperature (T=1.5) sample from an early, barely-trained
  pretrain checkpoint (`ckpt_early.pt`, pinned at step 20 by
  `pretrain.early_ckpt_step`).

That makes the preference trivially learnable: DPO loss falls from ln 2 ≈ 0.693
to ~0.01 in twenty steps and reward accuracy pins at 100%. Those numbers say the
loss, the frozen reference, and the gradient path all work. They say nothing
about story quality. At 1–33M parameters there is not enough capability for
preference optimisation to improve anything.

What is real and worth having: the DPO loss against a frozen reference copy of
the SFT model, prompt-masked sequence log-probabilities, the reward
diagnostics (`reward_chosen`, `reward_rejected`, `reward_margin`,
`reward_accuracy`), and the tests that pin all of it down. The code path is
exercised here so it is trustworthy when it later runs on a model where
preferences matter.

---

## Evaluation and the judge

`eval.json` reports validation perplexity plus three judged axes. Perplexity is
only comparable between models sharing a tokenizer — the tokenizer hash in each
checkpoint is what lets you verify that.

The judge is **offline by default**. `StubJudge` returns fixed scores without
leaving the machine, so the pipeline runs with no API key and no cost; its
numbers are placeholders and mean nothing about the model.

The real judge is opt-in through an environment variable, because a config file
that silently starts spending money when someone re-runs a stage is a bad
default:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...            # or `ant auth login`
TINYSTORIES_JUDGE=api make evaluate     # TINYSTORIES_JUDGE_MODEL to pick a model
```

It asks Claude for a 0–10 score per axis plus a one-line comment, constrained to
a JSON schema so the response parses deterministically. Held-out prompts live in
`src/tinystories_lm/eval/prompts.py` — written by hand rather than sampled from
the corpus, so they cannot have leaked into the training split. Base checkpoints
are judged on story-completion prompts and post-SFT checkpoints on
instruction-style prompts.

---

## Scaling up

The next milestone runs this same code on multi-GPU AWS by **swapping config
files only**. What is already in place:

- one device abstraction (`utils/device.py`) — nothing else in the repo calls
  `torch.cuda.is_available()`;
- gradient accumulation, so the global batch is a config choice independent of
  what fits on one device;
- `pretrain.stop_at_step` + `resume: auto`, which is what a preempted spot
  instance needs;
- a data sampler that already shards by `rank`/`world_size` and is correct at
  `world_size=1`;
- fused AdamW on CUDA, bf16 autocast selected by device capability, and
  `torch.compile` behind a flag.

`configs/pretrain/full.yaml` is the file that changes. What is deliberately
*not* built yet, marked `# TODO(scale)` in the source: DDP/FSDP init and
gradient all-reduce, and the distributed sampler wiring.

Also out of scope for this milestone: multi-node, custom CUDA kernels, MoE,
long-context or attention variants, quantization, a web UI, Docker, CI, and a
learned reward model. No hyperparameter tuning was done.

---

## Repository layout

```
configs/                  every knob; the only thing that differs between a laptop
  model/                  and an 8xH100 run
  data/  pretrain/  sft/  dpo/  eval/
src/tinystories_lm/
  model/      config.py attention.py layers.py transformer.py
  data/       download.py tokenizer.py prepare.py loader.py
  pretrain/   train.py optim.py schedule.py
  posttrain/  sft.py dpo.py
  inference/  sampling.py generate.py
  eval/       perplexity.py judge.py prompts.py harness.py
  utils/      device.py checkpoint.py config.py logging.py seed.py
scripts/                  thin argparse wrappers, no logic
tests/                    the nine gates and everything around them
```

Two small departures from the brief's tree, both for legibility:
`eval/harness.py` holds the eval stage (so `scripts/evaluate.py` stays a
wrapper), and stage config dataclasses live together in `utils/config.py` so
every knob in the pipeline is visible in one file.

---

## Dependencies

Deliberately minimal — the training loop, model, sampling, and data pipeline are
all written out here rather than imported.

| Package | Why |
|---|---|
| `torch` | model + hand-written training loop |
| `tokenizers` | BPE trainer; training byte-level BPE in pure Python would blow the smoke-test time budget |
| `numpy` | `uint16` memmap token streams |
| `pyyaml` | config files |
| `tqdm` | progress bars for download / tokenize / train |

Dataset download uses stdlib `urllib` (HTTP range requests against the
HuggingFace CDN), so `datasets` and `huggingface_hub` are **not** required.
`pytest` is a dev extra; `wandb` and `anthropic` are optional and only imported
when explicitly enabled.

Python 3.11+, PyTorch 2.x. Runs on CUDA, Apple Silicon (MPS), or CPU — resolved
by `utils/device.py`, overridable with `--<stage>.device=cpu`.

## License

The code is MIT. The TinyStories corpora are CDLA-Sharing-1.0 and are not
redistributed here — `prepare_data` downloads them at first run.
