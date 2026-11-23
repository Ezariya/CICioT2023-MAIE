# Makefile — convenience targets for the CICIoT2023 IDS pipeline.
# Run from the repo root. Each target maps to a step in README §4.

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

.PHONY: help install inspect ingest clean sample split train-binary \
        train-family train-finegrained feature-select benchmark simulate \
        dashboard test smoke-test clean-all

help:
	@echo "CICIoT2023 IDS — make targets"
	@echo "  install           pip install -r requirements.txt"
	@echo "  inspect           list CSV files + header check"
	@echo "  ingest            PySpark ingestion -> data/interim/ + metadata"
	@echo "  clean             cleaning + leakage audit"
	@echo "  sample            sample datasets A B C D"
	@echo "  split             stratified train/val/test split"
	@echo "  train-binary      train all 4 binary models on Dataset B"
	@echo "  train-family      train 8-class RF on Dataset C"
	@echo "  train-finegrained train 28-class classifier on Dataset D"
	@echo "  feature-select    compute feature importance + top-k sets"
	@echo "  benchmark         edge benchmark on PC (100/1000/10000 rows)"
	@echo "  simulate          generate synthetic pcap + run live IDS"
	@echo "  dashboard         launch Streamlit dashboard"
	@echo "  test              run pytest"
	@echo "  smoke-test        end-to-end smoke test on synthetic data"

install:
	$(PIP) install -r requirements.txt

inspect:
	PYTHONPATH=. $(PYTHON) -m src.data.inspect

ingest:
	PYTHONPATH=. $(PYTHON) -m src.data.ingest

clean:
	PYTHONPATH=. $(PYTHON) -m src.data.clean

sample:
	PYTHONPATH=. $(PYTHON) -m src.data.sample --dataset all

split:
	PYTHONPATH=. $(PYTHON) -m src.data.split --dataset B --task binary
	PYTHONPATH=. $(PYTHON) -m src.data.split --dataset C --task family
	PYTHONPATH=. $(PYTHON) -m src.data.split --dataset D --task finegrained

train-binary:
	PYTHONPATH=. $(PYTHON) -m src.models.train --task binary \
	    --model LogisticRegression,DecisionTree,RandomForest,XGBoost,LightGBM \
	    --dataset B

train-family:
	PYTHONPATH=. $(PYTHON) -m src.models.train --task family \
	    --model RandomForest,XGBoost,LightGBM --dataset C

train-finegrained:
	PYTHONPATH=. $(PYTHON) -m src.models.train --task finegrained \
	    --model RandomForest,XGBoost --dataset D

feature-select:
	PYTHONPATH=. $(PYTHON) -m src.features.selection

benchmark:
	PYTHONPATH=. $(PYTHON) -m src.evaluation.benchmark --mode edge \
	    --model-dir models/binary/XGBoost --rows 100
	PYTHONPATH=. $(PYTHON) -m src.evaluation.benchmark --mode edge \
	    --model-dir models/binary/XGBoost --rows 1000
	PYTHONPATH=. $(PYTHON) -m src.evaluation.benchmark --mode edge \
	    --model-dir models/binary/XGBoost --rows 10000

simulate:
	PYTHONPATH=. $(PYTHON) -m src.simulation.simulator \
	    --model-dir models/binary/XGBoost --generate

dashboard:
	PYTHONPATH=. streamlit run src/dashboard/app.py

test:
	PYTHONPATH=. $(PYTEST) tests/ -v

# End-to-end smoke test on synthetic data
smoke-test: install
	PYTHONPATH=. $(PYTHON) /home/z/my-project/scripts/make_synthetic_ciciot2023.py --rows-per-class 100
	PYTHONPATH=. $(PYTHON) -m src.data.inspect
	PYTHONPATH=. $(PYTHON) -m src.data.ingest
	PYTHONPATH=. $(PYTHON) -m src.data.clean
	PYTHONPATH=. $(PYTHON) -m src.data.sample --dataset B
	PYTHONPATH=. $(PYTHON) -m src.data.split --dataset B --task binary
	PYTHONPATH=. $(PYTHON) -m src.models.train --task binary --model LogisticRegression,RandomForest,XGBoost,LightGBM --dataset B
	PYTHONPATH=. $(PYTHON) -m src.inference.predictor --model-dir models/binary/XGBoost --export
	PYTHONPATH=. $(PYTHON) -m src.simulation.simulator --model-dir models/binary/XGBoost --generate
	PYTHONPATH=. $(PYTEST) tests/ -v

clean-all:
	rm -rf data/interim/* data/processed/*/*.parquet data/metadata/*.json
	rm -rf models/*/*.joblib models/*/*.json
	rm -rf reports/*.json reports/*.csv reports/*.png
	rm -rf simulation/traffic/*.pcap simulation/traffic/*.log
