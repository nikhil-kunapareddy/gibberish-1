# gibberish-1

A small language model built from scratch — tokenizer, architecture, training
loop and sampling written out rather than imported.

Early days: this repo is just getting started, so the structure below is intent,
not yet code.

## Plan

- **Tokenizer** — trained from scratch (byte-level BPE) rather than a
  pretrained vocab.
- **Model** — a decoder-only transformer, written out layer by layer.
- **Training** — a single readable loop, checkpointed and resumable, sized to
  train on local hardware.
- **Sampling** — temperature / top-k / top-p, plus whatever is useful for
  poking at what the model actually learned.

## Status

Nothing trained yet.
