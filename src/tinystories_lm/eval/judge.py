"""LLM-as-judge, scoring completions on grammar, creativity and consistency.

The three-axis decomposition is the TinyStories paper's, and it is worth
keeping: a 1M-parameter model's output is uniformly bad on a single "quality"
score, but it is legible when you can see that grammar is improving while
consistency has not started to.

Two judges ship here:

* `StubJudge` (default) -- returns fixed scores without leaving the machine, so
  `make smoke` runs with no API key, no network, and no cost. Its scores are
  deterministic placeholders and mean nothing about the model.
* `ClaudeJudge` -- a real Claude call, opt-in. Enable it with the environment
  variable `TINYSTORIES_JUDGE=api` (or `eval.judge=api` in the config); it needs
  the `anthropic` package and credentials in the environment.

The opt-in is deliberately an environment variable: a config file that silently
starts spending money when someone re-runs a stage is a bad default.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Protocol

AXES = ("grammar", "creativity", "consistency")

JUDGE_SYSTEM = (
    "You are grading text completions produced by a very small language model "
    "(1M-33M parameters) trained only on simple children's stories. Grade "
    "generously relative to that scale: the question is whether the model has "
    "learned the property at all, not whether the writing is good."
)

JUDGE_TEMPLATE = """Here is the beginning of a story, and a completion written by a student model.

--- STORY BEGINNING ---
{prompt}
--- COMPLETION ---
{completion}
--- END ---

Score the completion from 0 to 10 on each axis:
- grammar: is it well-formed English? Are words real words, sentences complete?
- creativity: does it introduce anything beyond copying the prompt's words?
- consistency: does it follow from the beginning without contradicting it?

Then give a one-sentence comment."""

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "grammar": {"type": "integer", "minimum": 0, "maximum": 10},
        "creativity": {"type": "integer", "minimum": 0, "maximum": 10},
        "consistency": {"type": "integer", "minimum": 0, "maximum": 10},
        "comment": {"type": "string"},
    },
    "required": ["grammar", "creativity", "consistency", "comment"],
    "additionalProperties": False,
}


@dataclass
class Score:
    grammar: float
    creativity: float
    consistency: float
    comment: str = ""
    prompt: str = ""
    completion: str = ""

    def to_dict(self) -> dict:
        return {
            "grammar": self.grammar,
            "creativity": self.creativity,
            "consistency": self.consistency,
            "comment": self.comment,
            "prompt": self.prompt,
            "completion": self.completion,
        }


class Judge(Protocol):
    name: str

    def score(self, prompt: str, completion: str) -> Score: ...


@dataclass
class StubJudge:
    """Offline judge: fixed scores, no network, no API key, no cost.

    Returns the same numbers for every completion. It exists so the pipeline
    has a working judge stage by default -- do not read anything into its
    output beyond "the stage ran".
    """

    name: str = "stub"
    fixed: tuple[float, float, float] = (3.0, 2.0, 2.0)

    def score(self, prompt: str, completion: str) -> Score:
        return Score(
            grammar=self.fixed[0],
            creativity=self.fixed[1],
            consistency=self.fixed[2],
            comment="offline stub judge: fixed scores, not a real evaluation",
            prompt=prompt,
            completion=completion,
        )


@dataclass
class ClaudeJudge:
    """Real judge: one Claude call per completion, structured JSON back."""

    model: str = "claude-opus-5"
    max_tokens: int = 1000
    name: str = field(init=False, default="claude")

    def __post_init__(self) -> None:
        try:
            import anthropic  # noqa: PLC0415 - optional, only for the opt-in judge
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "the API judge needs the `anthropic` package: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic()
        self.name = f"claude:{self.model}"

    def score(self, prompt: str, completion: str) -> Score:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=JUDGE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_TEMPLATE.format(prompt=prompt, completion=completion),
                }
            ],
            # Low effort: grading a 60-token story is not a reasoning problem,
            # and the judge runs once per prompt.
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": SCORE_SCHEMA},
            },
        )
        # output_config.format guarantees the first text block is valid JSON.
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return Score(
            grammar=float(data["grammar"]),
            creativity=float(data["creativity"]),
            consistency=float(data["consistency"]),
            comment=data.get("comment", ""),
            prompt=prompt,
            completion=completion,
        )


def build_judge(kind: str = "stub", model: str = "claude-opus-5") -> Judge:
    """Resolve the judge, letting `TINYSTORIES_JUDGE` override the config.

    `TINYSTORIES_JUDGE=api` turns on the real judge; `=stub` forces it off even
    if a config asks for it (useful in CI).
    """
    kind = os.environ.get("TINYSTORIES_JUDGE", kind).strip().lower()
    if kind == "stub":
        return StubJudge()
    if kind == "api":
        return ClaudeJudge(model=os.environ.get("TINYSTORIES_JUDGE_MODEL", model))
    raise ValueError(f"unknown judge {kind!r}; expected stub|api")


def aggregate(scores: list[Score]) -> dict:
    """Mean per axis, plus the overall mean the paper reports."""
    if not scores:
        return {axis: float("nan") for axis in AXES}
    out = {axis: statistics.fmean(getattr(s, axis) for s in scores) for axis in AXES}
    out["overall"] = statistics.fmean(out[axis] for axis in AXES)
    out["n"] = len(scores)
    return out
