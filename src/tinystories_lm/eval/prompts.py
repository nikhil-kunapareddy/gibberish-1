"""Held-out story-completion prompts for qualitative evaluation.

These are written here rather than sampled from the corpus so that they are
genuinely held out: no prompt below appears in TinyStories, and none can have
leaked into the training split. They follow the paper's structure -- a couple of
sentences that set up a situation and hand the model an obvious continuation.
"""

from __future__ import annotations

COMPLETION_PROMPTS: list[str] = [
    "Once upon a time there was a little girl named Mia who found a red balloon in the park.",
    "Tom had a small blue car. One day the car would not move, so Tom",
    "Lily and Ben went to the beach. They wanted to build the biggest sandcastle, but",
    "The old dog was tired. He wanted to sleep, but a noisy bird kept",
    "Sara opened the box. Inside she saw something that made her eyes go wide:",
    "It was raining very hard. The cat did not like the rain, so she",
    "Max found a key under the tree. He wondered what it opened, and then he",
    "The little robot could not find his friend. He looked everywhere until",
]

# Instruction-style prompts matching the TinyStories-Instruct format, for
# evaluating a model after SFT. The trailing `Story:` is the handoff marker the
# SFT data uses.
INSTRUCT_PROMPTS: list[str] = [
    "Features: Dialogue\nWords: kite, windy, laugh\nSummary: Two friends fly a kite on a windy day.\nStory:\n",
    "Words: soup, spoon, warm\nSummary: A boy makes soup for his sick sister.\nStory:\n",
    "Features: BadEnding\nSummary: A cat chases a mouse and gets stuck in a box.\nStory:\n",
    "Random sentence: The door opened all by itself.\nSummary: A girl explores an empty house.\nStory:\n",
    "Features: Dialogue, MoralValue\nWords: share, apple, friend\nSummary: Two children learn to share one apple.\nStory:\n",
]


def get_prompts(kind: str = "completion", n: int | None = None) -> list[str]:
    """`kind` is "completion" (base models) or "instruct" (post-SFT models)."""
    pool = {"completion": COMPLETION_PROMPTS, "instruct": INSTRUCT_PROMPTS}.get(kind)
    if pool is None:
        raise ValueError(f"unknown prompt kind {kind!r}; expected completion|instruct")
    return pool[:n] if n else list(pool)
