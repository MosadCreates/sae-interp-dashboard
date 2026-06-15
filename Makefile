.PHONY: collect train analyze api dashboard all clean

COLLECT_CONFIG ?= configs/collect.yaml
TRAIN_CONFIG  ?= configs/train_medium.yaml
ANALYZE_CONFIG ?= configs/analyze.yaml
API_CONFIG    ?= configs/api.yaml

# --- Python targets ---

collect:
	python -m src.data.collect_activations --config $(COLLECT_CONFIG)

train:
	python -m src.sae.train --config $(TRAIN_CONFIG)

analyze:
	python -m src.analysis.pipeline --config $(ANALYZE_CONFIG)

api:
	uvicorn src.api.main:app --reload --port 8000

# --- Dashboard ---

dashboard:
	cd dashboard && npm run dev

# --- Full pipeline ---

all: collect train analyze

# --- Utilities ---

clean:
	rm -rf data/activations.h5 models/*/ results/*.json results/geometry/ __pycache__ */__pycache__ */**/__pycache__

test:
	python -m pytest tests/ -v

lint:
	python -m flake8 src/
	cd dashboard && npm run lint
