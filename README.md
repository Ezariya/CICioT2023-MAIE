# CICIoT2023 AI-Based IoT IDS

An AI-based Intrusion Detection System for IoT / embedded networks built on the
official **CICIoT2023** dataset (UNB, 2024). The project has two strictly
separated environments:

| Environment | Purpose | Hardware |
| --- | --- | --- |
| **Offline training** | PySpark ingestion of 46.7M flow records, cleaning, sampling, model training, evaluation, export | Main workstation |
| **Online / simulation** | Docker / VM simulating an IoT network, live traffic capture, flow extraction, ML inference, alerts, dashboard | VM and / or Raspberry Pi 4B |

---

## 1. Research question

> How can machine learning be used to detect and classify cyber attacks in IoT
> network traffic while maintaining sufficiently low computational and inference
> costs for deployment on resource-constrained edge devices?

Secondary questions (mapped to experiments):

1. How accurately can attacks be detected?
2. Which ML model provides the best accuracy / performance trade-off?
3. How does severe class imbalance affect detection?
4. Which features are most useful?
5. Can the final model run efficiently on a Raspberry Pi?
6. What is the inference latency on edge hardware?
7. Can the trained model detect attacks in a simulated network environment?

---

## 2. Repository layout

```
ciciot2023-ids/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── config.yaml              # global config (spark, splits, models, sim, edge)
│   └── features.yaml            # feature schema + leakage audit rules
│
├── data/
│   ├── raw/CICIoT2023/          # official CSV release goes here (NOT committed)
│   ├── interim/                 # cleaned parquet
│   ├── processed/{binary,multiclass,edge}/   # train/val/test parquet
│   └── metadata/                # dataset stats, leakage audit, sampling logs
│
├── notebooks/                   # EDA, baselines, model comparison, edge eval
├── src/
│   ├── data/         # Spark ingestion, cleaning, sampling, splitting
│   ├── features/     # schema, preprocessing, feature selection
│   ├── models/       # baseline, RF, XGBoost/LightGBM, train driver
│   ├── evaluation/   # metrics, confusion, edge/PC benchmark
│   ├── inference/    # model loader, predictor (no retraining)
│   └── simulation/   # traffic capture, flow extraction, simulator, alerting
│
├── models/{binary,multiclass,edge}/   # model.joblib + metadata.json
├── simulation/{docker,configs,traffic}/
├── deployment/raspberry_pi/     # Pi setup, systemd, install scripts
├── reports/                     # EDA report, leakage audit, comparison tables
└── tests/                       # unit tests for schema, sampler, predictor
```

Training, inference, and simulation are kept separate by design.

---

## 3. Setup

```bash
git clone <this repo>
cd ciciot2023-ids
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Spark**: a local `pyspark` install is enough for development. For the full
46M-row corpus, point `config.spark.master` at a YARN / K8s cluster and raise
`driver_memory` / `executor_memory`.

**Dataset**: download the **CSV** release (not PCAP for V1) from the official
UNB page <https://www.unb.ca/cic/datasets/iotdataset-2023.html> and unzip the
CSV files into `data/raw/CICIoT2023/`.

---

## 4. Development order

The 24-step development order from the project plan maps to these commands.
Every step writes a JSON / parquet artefact under `data/` or `models/` so it
is reproducible.

| Step | Module | Command |
| --- | --- | --- |
| 01 | repo + config | `pip install -r requirements.txt` |
| 02 | Spark ingestion | `python -m src.data.ingest` |
| 03 | dataset stats | `python -m src.data.inspect` |
| 04 | data quality + leakage audit | `python -m src.data.clean` |
| 05 | reproducible sampling | `python -m src.data.sample --dataset A` |
| 06 | train/val/test split | `python -m src.data.split --dataset A` |
| 07 | binary baseline | `python -m src.models.train --task binary --model LogisticRegression` |
| 08 | binary model comparison | `python -m src.models.train --task binary --model RandomForest,XGBoost,LightGBM` |
| 09 | 8-class classifier | `python -m src.models.train --task family` |
| 10 | 34-class classifier | `python -m src.models.train --task finegrained` |
| 11 | feature importance | `python -m src.features.selection --method random_forest_importance` |
| 12 | feature reduction | `python -m src.features.selection --top-k 20,15,10` |
| 13 | model optimization | `python -m src.evaluation.benchmark --mode model` |
| 14 | export inference model | `python -m src.inference.predictor --export` |
| 15 | isolated network sim | `cd simulation/docker && docker compose up -d` |
| 16 | capture simulated traffic | `python -m src.simulation.traffic_capture --duration 60` |
| 17 | flow / feature extraction | `python -m src.simulation.flow_extraction --pcap ...` |
| 18 | connect extraction → ML | `python -m src.simulation.simulator --with-model` |
| 19 | real-time detection | `python -m src.simulation.simulator --live` |
| 20 | Raspberry Pi deployment | see `deployment/raspberry_pi/README.md` |
| 21 | edge benchmarking | `python -m src.evaluation.benchmark --mode edge` |
| 22 | dashboard + alerts | `streamlit run src/dashboard/app.py` |
| 23 | final experiments | `python -m src.evaluation.run_experiments` |
| 24 | documentation + presentation | `reports/` |

---

## 5. Hard constraints (enforced everywhere)

These constraints are non-negotiable and are checked by `tests/`:

- **Never** load the full 46M-row corpus into a single pandas DataFrame.
  Use PySpark for ingestion / cleaning / sampling / statistics.
- **Never** compute scaling / normalisation statistics on the full dataset
  before the train/val/test split. Fit preprocessing on TRAIN only.
- **Never** use accuracy as the only metric. Always report precision, recall,
  macro-F1, weighted-F1, balanced accuracy, FPR, FNR, confusion matrix.
- **Never** silently drop columns. Each dropped column has a documented
  reason in `config/features.yaml:leakage_audit` and in
  `reports/leakage_audit.json`.
- **Never** replay a CSV row and call that live detection. The simulation
  must capture real packets → compute flow features → run the model.
- **Never** train the model on the Raspberry Pi. The Pi only loads the
  exported `model.joblib` and runs inference.
- **Never** direct any attack traffic at external networks. The simulation
  runs in an isolated Docker network only.

---

## 6. Datasets, tasks, models

### Datasets (reproducible, `seed: 42`)

| ID | Purpose | Target rows |
| --- | --- | --- |
| A | original distribution sample | ~2,000,000 |
| B | balanced binary | 1,000,000 (500k Benign + 500k Attack) |
| C | stratified 8-class | ~640,000 (80k per family) |
| D | edge-lightweight | ~200,000 |

### ML tasks

| Task | Classes | Question |
| --- | --- | --- |
| **binary** | Benign / Attack | Is traffic malicious? (deployment model) |
| **family** | 8 (Benign + 7 attack families) | Which family? |
| **finegrained** | 28 individual attacks | Which specific attack? (academic) |

### Models compared

| Model | Macro-F1 | Recall | FPR | Latency | Model size |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | _filled at runtime_ | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |
| XGBoost / LightGBM | | | | | |

The model with the best **accuracy/performance trade-off** is selected — not
the one with the highest accuracy.

---

## 7. Citation

```
E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, J. Lu, A. A. Ghorbani,
"CICIoT2023: A real-time dataset and benchmark for IoT intrusion detection",
Elsevier Internet of Things, 2024.  DOI: 10.1016/j.iot.2024.101014
```

Dataset: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>

---

## 8. License & safety

This project is for **educational / research** use only. Attack traffic must
be generated only inside the isolated Docker network defined in
`simulation/docker/`. Never direct attack tools at networks you do not own.
