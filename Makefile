# tinystories-lm
#
# Every target is a thin wrapper over scripts/. Nothing here knows anything the
# config files do not -- `make smoke` is just the smoke configs.

PYTHON  ?= python3
export PYTHONPATH := $(CURDIR)/src

DATA     ?= configs/data/smoke.yaml
MODEL    ?= configs/model/smoke-1m.yaml
PRETRAIN ?= configs/pretrain/smoke.yaml
SFT      ?= configs/sft/smoke.yaml
DPO      ?= configs/dpo/smoke.yaml
EVAL     ?= configs/eval/smoke.yaml
CKPT     ?= runs/pretrain-smoke/ckpt_best.pt
DEVICE   ?= auto
ARGS     ?=

VENV     ?= .venv
KERNEL   ?= tinystories-lm

.PHONY: help smoke test test-fast data pretrain sft dpo evaluate generate profile kernel clean clean-runs clean-all

help:            ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

smoke:           ## Full pipeline + all nine gates, with a timing table (the deliverable)
	$(PYTHON) scripts/smoke_test.py --device $(DEVICE) $(ARGS)

test:            ## Run the whole test suite, including the slow gates
	$(PYTHON) -m pytest tests -v

test-fast:       ## Skip the gates that train for a few hundred steps
	$(PYTHON) -m pytest tests -q -m "not slow"

data:            ## Stage 1: download + tokenize + split
	$(PYTHON) scripts/prepare_data.py --data $(DATA)

pretrain:        ## Stage 2: pretrain
	$(PYTHON) scripts/pretrain.py --pretrain $(PRETRAIN) --model $(MODEL) $(ARGS)

sft:             ## Stage 3: instruction tuning
	$(PYTHON) scripts/sft.py --sft $(SFT) $(ARGS)

dpo:             ## Stage 4: DPO (mechanics only at this scale)
	$(PYTHON) scripts/dpo.py --dpo $(DPO) $(ARGS)

evaluate:        ## Stage 5: perplexity + judge
	$(PYTHON) scripts/evaluate.py --eval $(EVAL) $(ARGS)

generate:        ## Sample from a checkpoint: make generate CKPT=... PROMPT="..."
	$(PYTHON) scripts/generate.py --checkpoint $(CKPT) $(if $(PROMPT),--prompt "$(PROMPT)",)

profile:         ## Tokens/sec with and without the KV cache
	$(PYTHON) scripts/generate.py --checkpoint $(CKPT) --profile

kernel:          ## Create .venv with the project and register it as Jupyter kernel "tinystories-lm"
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -q --upgrade pip
	$(VENV)/bin/python -m pip install -q -e ".[dev,notebook]"
	$(VENV)/bin/python -m ipykernel install --user --name $(KERNEL) --display-name "Python ($(KERNEL))"

clean:           ## Remove run outputs (keeps the downloaded corpus and prepared data)
	rm -rf runs/ .pytest_cache/
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

clean-runs: clean

clean-all: clean  ## Also remove prepared data and the raw download cache
	rm -rf data/
