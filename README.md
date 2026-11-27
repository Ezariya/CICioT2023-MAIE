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
git clone https://github.com/Ezariya/CICioT2023-MAIE.git
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
### binary Classification Results

| Model | Accuracy | Precision | Recall | F1 Macro | F1 Weighted | Balanced Acc | FPR | FNR | Train (s) | Latency Mean (ms) | Latency P95 (ms) | Throughput (FPS) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression** | 0.8356 | 0.9870 | 0.7958 | 0.8073 | 0.8466 | 0.8808 | 0.0342 | 0.2042 | 9.8855 | 0.5163 | 0.6695 | 7,170,751 | 0.0023 |
| **DecisionTree** | 0.9031 | 0.9832 | 0.8886 | 0.8773 | 0.9072 | 0.9195 | 0.0496 | 0.1114 | 16.3757 | 0.1485 | 0.1778 | 4,418,963 | 1.2999 |
| **RandomForest** | 0.9083 | 0.9845 | 0.8944 | 0.8835 | 0.9121 | 0.9241 | 0.0461 | 0.1056 | 143.6791 | 9.7853 | 11.2220 | 57,409 | 87.6809 |
| **XGBoost** | 0.9129 | 0.9878 | 0.8973 | 0.8893 | 0.9164 | 0.9306 | 0.0361 | 0.1027 | 11.4809 | 0.6361 | 0.7214 | 102,473 | 3.7028 |
| **LightGBM** | 0.9125 | 0.9871 | 0.8975 | 0.8887 | 0.9161 | 0.9295 | 0.0384 | 0.1025 | 42.9734 | 1.4977 | 1.6469 | 26,517 | 2.6202 |
| **PyTorch_ANN** | 0.9016 | 0.9365 | 0.9349 | 0.8630 | 0.9017 | 0.8637 | 0.2076 | 0.0651 | 111.5261 | 1.0053 | 2.3616 | 5,024,120 | 0.0615 |
| **PyTorch_CNN1D** | 0.8991 | 0.9475 | 0.9192 | 0.8638 | 0.9006 | 0.8762 | 0.1667 | 0.0808 | 115.3179 | 0.8858 | 1.0502 | 1,845,961 | 0.0787 |



The model with the best **accuracy/performance trade-off** is selected — not
the one with the highest accuracy.

> **Best trade-off model (family):** XGBoost

### Family Classification Results

| Model | Accuracy | Precision | Recall | F1 Macro | F1 Weighted | Balanced Acc | FPR | FNR | Train (s) | Latency Mean (ms) | Latency P95 (ms) | Throughput (FPS) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | 0.8006 | 0.6817 | 0.7311 | 0.6932 | 0.8099 | 0.7311 | 0.0289 | 0.2689 | 172.4014 | 9.7227 | 11.8688 | 49,310 | 286.0290 |
| **XGBoost** | 0.8419 | 0.7504 | 0.7357 | 0.7412 | 0.8424 | 0.7357 | 0.0241 | 0.2643 | 52.2024 | 1.5776 | 1.7398 | 11,507 | 26.7876 |
| **LightGBM** | 0.7909 | 0.6724 | 0.7610 | 0.6808 | 0.8076 | 0.7610 | 0.0297 | 0.2390 | 94.8844 | 1.1252 | 1.2474 | 12,351 | 4.7257 |
| **PyTorch_ANN** | 0.8171 | 0.8447 | 0.6421 | 0.6694 | 0.8079 | 0.6421 | 0.0292 | 0.3579 | 111.6985 | 0.6590 | 0.9794 | 4,834,291 | 0.0632 |
| **PyTorch_CNN1D** | 0.8033 | 0.7806 | 0.6299 | 0.6533 | 0.7959 | 0.6299 | 0.0308 | 0.3701 | 116.9194 | 0.9329 | 1.4034 | 1,810,529 | 0.0804 |

> **Best trade-off model (family):** XGBoost

### Fine-Grained Classification Results

| Model | Accuracy | Precision | Recall | F1 Macro | F1 Weighted | Balanced Acc | FPR | FNR | Train (s) | Latency Mean (ms) | Latency P95 (ms) | Throughput (FPS) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | 0.7022 | 0.5965 | 0.6121 | 0.5829 | 0.7215 | 0.6121 | 0.0091 | 0.3879 | 103.1248 | 8.9231 | 14.2722 | 71,771 | 257.5776 |
| **XGBoost** | 0.7756 | 0.6297 | 0.6208 | 0.6196 | 0.7684 | 0.6208 | 0.0072 | 0.3792 | 212.5867 | 5.9355 | 7.4013 | 2,178 | 75.5437 |
| **PyTorch_ANN** | 0.7344 | 0.6030 | 0.5517 | 0.5502 | 0.7004 | 0.5517 | 0.0089 | 0.4483 | 107.7555 | 0.8966 | 1.2955 | 3,656,692 | 0.0697 |
| **PyTorch_CNN1D** | 0.7354 | 0.6226 | 0.5489 | 0.5526 | 0.7008 | 0.5489 | 0.0090 | 0.4511 | 114.1172 | 0.8082 | 0.9053 | 1,582,808 | 0.0868 |

> **Best trade-off model (fine-grained):** XGBoost


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
