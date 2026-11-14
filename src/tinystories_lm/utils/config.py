"""YAML configs + dotted CLI overrides.

One rule holds the whole repo together: **a run is defined by its config
files**. Nothing is hardcoded in a script, and every knob is reachable from the
command line as `--<section>.<field>=<value>`:

    python scripts/pretrain.py --pretrain configs/pretrain/smoke.yaml \
        --model configs/model/smoke-1m.yaml --pretrain.lr=3e-4

That is the property the AWS scale-up depends on: the difference between the
laptop run and the 8xH100 run should be which YAML files you point at, not a
diff in `src/`.

Sections are typed dataclasses (below), so a typo in a YAML key is an error at
load time rather than a silently ignored setting.
"""

from __future__ import annotations

import dataclasses
import json
import types
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Type, Union, get_args, get_origin

import yaml

from ..model.config import ModelConfig

# --------------------------------------------------------------------------- #
# Stage configs
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """`prepare_data` inputs: what to download, how to tokenize, how to split."""

    name: str = "smoke"
    data_dir: str = "data"  # raw download cache (shared across runs)
    out_dir: str = "data/smoke"  # prepared .bin + meta.json live here

    dataset: str = "roneneldan/TinyStories"
    instruct_dataset: str = "roneneldan/TinyStories-Instruct"
    # Which upstream file to read. The -valid splits are ~19MB/27MB and hold
    # ~22k/~22k records, plenty for a smoke run; the -train splits are ~1.9GB.
    source_file: str = "TinyStories-valid.txt"
    instruct_source_file: str = "TinyStories-Instruct-valid.txt"
    # Range-download only the first N bytes of each source (None = whole file).
    # Keeps `make smoke` from pulling gigabytes on a laptop.
    max_download_bytes: int | None = 24_000_000

    num_stories: int | None = 5000  # None = every story in the source
    num_instruct_examples: int | None = 2000
    tokenizer_train_stories: int = 5000  # cap on BPE training corpus size

    vocab_size: int = 8192
    val_fraction: float = 0.05
    split_seed: int = 1234  # salts the story hash -> split assignment

    def validate(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (byte-level BPE needs the byte alphabet)")


@dataclass
class PretrainConfig:
    """The pretraining loop."""

    name: str = "pretrain-smoke"
    data_dir: str = "data/smoke"  # must contain train.bin/val.bin/meta.json
    out_dir: str = "runs/pretrain-smoke"

    max_steps: int = 200
    batch_size: int = 16
    block_size: int = 256
    grad_accum_steps: int = 1

    lr: float = 1e-3
    min_lr_ratio: float = 0.1  # cosine floor, as a fraction of lr
    warmup_steps: int = 20
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    eval_interval: int = 50
    eval_iters: int = 20
    log_interval: int = 10
    ckpt_interval: int = 100
    keep_last_k: int = 2
    # An early, barely-trained checkpoint is what DPO samples its "rejected"
    # continuations from, so pin one deliberately instead of hoping one survives
    # the keep-last-k rotation.
    early_ckpt_step: int = 20

    device: str = "auto"
    amp: str | bool = "auto"
    compile: bool = False  # off by default: slow warmup, flaky on MPS
    seed: int = 1337
    resume: str = ""  # "" | "auto" | path to a checkpoint
    # Stop before max_steps *without* changing the LR schedule, then pick up
    # again with `resume`. This is what a preempted spot instance looks like,
    # and it is how the resume-equivalence gate interrupts a run.
    stop_at_step: int = 0  # 0 = run to max_steps

    wandb: bool = False
    wandb_project: str = "tinystories-lm"

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) exceeds max_steps ({self.max_steps})"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")


@dataclass
class SFTConfig:
    """Instruction tuning on TinyStories-Instruct, loss masked to the completion."""

    name: str = "sft-smoke"
    data_dir: str = "data/smoke"  # needs sft_train.jsonl / sft_val.jsonl
    out_dir: str = "runs/sft-smoke"
    init_from: str = "runs/pretrain-smoke/ckpt_best.pt"

    max_steps: int = 50
    batch_size: int = 8
    max_seq_len: int = 256
    grad_accum_steps: int = 1

    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    eval_interval: int = 25
    eval_iters: int = 10
    log_interval: int = 10
    ckpt_interval: int = 50
    keep_last_k: int = 2

    device: str = "auto"
    amp: str | bool = "auto"
    compile: bool = False
    seed: int = 1337
    resume: str = ""

    wandb: bool = False
    wandb_project: str = "tinystories-lm"

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_seq_len <= 1:
            raise ValueError("max_seq_len must be > 1")


@dataclass
class DPOConfig:
    """DPO against a frozen reference copy of the SFT model.

    At 1-33M params this cannot make the model better; it is here so the code
    path is exercised. See README "What DPO is doing here".
    """

    name: str = "dpo-smoke"
    data_dir: str = "data/smoke"
    out_dir: str = "runs/dpo-smoke"
    init_from: str = "runs/sft-smoke/ckpt_best.pt"  # policy AND frozen reference
    # Checkpoint that generates the "rejected" samples. An early, undertrained
    # checkpoint makes the preference signal trivially learnable.
    sampler_ckpt: str = "runs/pretrain-smoke/ckpt_early.pt"
    # Synthesised pairs live with the run, not with the corpus: they depend on
    # `sampler_ckpt`, so they must not outlive the checkpoints `make clean` removes.
    pairs_file: str = "runs/dpo-smoke/dpo_pairs.jsonl"

    num_pairs: int = 64
    pair_prompt_tokens: int = 32  # prompt length carved off the real story
    pair_max_new_tokens: int = 64
    pair_temperature: float = 1.5  # high temp -> visibly worse "rejected" text
    pair_top_k: int = 0

    beta: float = 0.1
    label_smoothing: float = 0.0

    max_steps: int = 20
    batch_size: int = 4
    max_seq_len: int = 256
    grad_accum_steps: int = 1

    lr: float = 5e-6
    min_lr_ratio: float = 1.0  # flat LR: DPO on 20 steps has nothing to decay
    warmup_steps: int = 2
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    log_interval: int = 5
    eval_interval: int = 10
    eval_iters: int = 5
    ckpt_interval: int = 20
    keep_last_k: int = 1

    device: str = "auto"
    amp: str | bool = "auto"
    compile: bool = False
    seed: int = 1337
    resume: str = ""

    wandb: bool = False
    wandb_project: str = "tinystories-lm"

    def validate(self) -> None:
        if self.beta <= 0:
            raise ValueError("dpo beta must be positive")
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")


@dataclass
class EvalConfig:
    """Validation perplexity + the paper's three-axis LLM judge."""

    name: str = "eval-smoke"
    data_dir: str = "data/smoke"
    checkpoint: str = "runs/pretrain-smoke/ckpt_best.pt"
    out_dir: str = "runs/eval-smoke"

    batch_size: int = 8
    block_size: int = 256
    max_batches: int = 50  # cap so eval stays inside the smoke budget

    # Judge: "stub" never leaves the machine and returns fixed scores, so the
    # pipeline runs with no API key. "api" is opt-in and additionally requires
    # TINYSTORIES_JUDGE_ENABLE=1 in the environment.
    judge: str = "stub"
    judge_model: str = "claude-opus-5"
    judge_prompts: int = 5
    judge_max_new_tokens: int = 64
    judge_temperature: float = 0.8
    judge_top_p: float = 0.95

    device: str = "auto"
    amp: str | bool = "auto"
    seed: int = 1337

    def validate(self) -> None:
        if self.judge not in ("stub", "api"):
            raise ValueError(f"judge must be 'stub' or 'api', got {self.judge!r}")


@dataclass
class GenerateConfig:
    """Sampling knobs for `scripts/generate.py`."""

    checkpoint: str = "runs/pretrain-smoke/ckpt_best.pt"
    data_dir: str = "data/smoke"  # for the tokenizer
    prompt: str = "Once upon a time"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    greedy: bool = False
    num_samples: int = 1
    use_cache: bool = True
    profile: bool = False
    device: str = "auto"
    seed: int = 1337

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


SECTIONS: dict[str, Type] = {
    "model": ModelConfig,
    "data": DataConfig,
    "pretrain": PretrainConfig,
    "sft": SFTConfig,
    "dpo": DPOConfig,
    "eval": EvalConfig,
    "generate": GenerateConfig,
}


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #


class Config(dict):
    """A dict of {section name: dataclass instance} with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"config section {name!r} was not loaded (have: {sorted(self)})"
            ) from exc

    def to_dict(self) -> dict:
        return {k: asdict(v) if is_dataclass(v) else v for k, v in self.items()}

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _coerce(value: Any, typ: Any, path: str) -> Any:
    """Coerce a YAML/CLI scalar to the dataclass field's annotated type."""
    origin = get_origin(typ)
    if origin is Union or origin is types.UnionType:  # `Union[X, None]` and `X | None`
        args = [a for a in get_args(typ) if a is not type(None)]
        if value is None:
            return None
        # Prefer an exact type match before falling back to coercion, so that
        # `amp: str | bool` keeps a YAML `false` a bool instead of turning it
        # into the string "False".
        args.sort(key=lambda a: a is not type(value))
        last_err: Exception | None = None
        for arg in args:
            try:
                return _coerce(value, arg, path)
            except (TypeError, ValueError) as e:
                last_err = e
        raise ValueError(f"{path}: cannot coerce {value!r} to {typ}") from last_err
    if typ is Any or typ is None:
        return value
    if typ is bool:
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{path}: cannot parse {value!r} as bool")
    if typ is int:
        if isinstance(value, bool):
            raise ValueError(f"{path}: expected int, got bool")
        if isinstance(value, float) and not float(value).is_integer():
            raise ValueError(f"{path}: expected int, got {value!r}")
        return int(value)
    if typ is float:
        return float(value)
    if typ is str:
        return str(value)
    if isinstance(typ, type) and isinstance(value, typ):
        return value
    raise ValueError(f"{path}: cannot coerce {value!r} to {typ}")


def build_section(cls: Type, values: Mapping[str, Any], section: str) -> Any:
    """Instantiate a section dataclass from a plain dict, typed and validated."""
    field_types = {f.name: f.type for f in fields(cls)}
    unknown = set(values) - set(field_types)
    if unknown:
        raise ValueError(
            f"unknown key(s) in [{section}]: {sorted(unknown)}. "
            f"Valid keys: {sorted(field_types)}"
        )
    kwargs = {}
    for key, raw in values.items():
        typ = field_types[key]
        if isinstance(typ, str):  # `from __future__ import annotations` stringifies these
            typ = _resolve_annotation(typ)
        kwargs[key] = _coerce(raw, typ, f"{section}.{key}")
    obj = cls(**kwargs)
    if hasattr(obj, "validate"):
        obj.validate()
    return obj


_ANNOTATION_MAP: dict[str, Any] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "int | None": Union[int, None],
    "str | None": Union[str, None],
    "float | None": Union[float, None],
    "str | bool": Union[str, bool],
}


def _resolve_annotation(annotation: str) -> Any:
    try:
        return _ANNOTATION_MAP[annotation.strip()]
    except KeyError as exc:
        raise ValueError(
            f"config field annotation {annotation!r} is not supported by the "
            f"loader; add it to _ANNOTATION_MAP"
        ) from exc


def parse_overrides(overrides: list[str]) -> dict[str, dict[str, Any]]:
    """Parse `--pretrain.lr=3e-4` / `pretrain.lr=3e-4` into {section: {field: value}}."""
    out: dict[str, dict[str, Any]] = {}
    for item in overrides:
        raw = item[2:] if item.startswith("--") else item
        if "=" not in raw:
            raise ValueError(f"override {item!r} must look like --section.field=value")
        key, value = raw.split("=", 1)
        if "." not in key:
            raise ValueError(f"override {item!r} must name a section, e.g. --pretrain.lr=3e-4")
        section, field_name = key.split(".", 1)
        if section not in SECTIONS:
            raise ValueError(
                f"override {item!r} names unknown section {section!r}; "
                f"known sections: {sorted(SECTIONS)}"
            )
        out.setdefault(section, {})[field_name] = _parse_scalar(value)
    return out


def _parse_scalar(value: str) -> Any:
    """Interpret an override value the same way YAML would."""
    if value == "":
        return ""
    if value.lower() in ("none", "null"):
        return None
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def build_config(paths: Mapping[str, str | None], overrides: list[str] | None = None) -> Config:
    """Load the named sections from YAML, then apply CLI overrides on top.

    `paths` maps section name -> YAML path (None means "use section defaults").
    Overrides may touch any loaded section; naming a section that was not
    loaded is an error, because silently ignoring `--model.n_layer=8` would be
    the worst possible failure mode.
    """
    overrides = overrides or []
    parsed = parse_overrides(overrides)

    raw: dict[str, dict[str, Any]] = {}
    for section, path in paths.items():
        if section not in SECTIONS:
            raise ValueError(f"unknown config section {section!r}")
        raw[section] = load_yaml(path) if path else {}

    for section, kv in parsed.items():
        if section not in raw:
            raise ValueError(
                f"override targets section {section!r}, but this command only "
                f"loads {sorted(raw)}"
            )
        raw[section].update(kv)

    cfg = Config()
    for section, values in raw.items():
        cfg[section] = build_section(SECTIONS[section], values, section)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=True, default_flow_style=False)


def config_field_names(section: str) -> list[str]:
    return [f.name for f in dataclasses.fields(SECTIONS[section])]
