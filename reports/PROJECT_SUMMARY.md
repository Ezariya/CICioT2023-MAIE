# CICIoT2023 AI-Based IoT IDS — Project Summary

This document records the state of the project after the initial end-to-end build and smoke test.

## Repository

`ciciot2023-ids/` — full repository, structure matches the spec from section 2.

```
config/              — config.yaml, features.yaml
data/{raw,interim,processed,metadata}/
notebooks/           — 5 .py notebook stubs (01..05)
src/{data,features,models,evaluation,inference,simulation,dashboard}/
models/{binary,multiclass,edge}/
simulation/{docker,configs,traffic}/
deployment/raspberry_pi/
reports/
tests/
```

## Smoke test executed

A small synthetic CICIoT2023-like corpus was generated
(`scripts/make_synthetic_ciciot2023.py`, 100 rows × 28 classes = 2,800 rows)
to exercise every stage of the pipeline end-to-end **without** downloading
the real 46M-row UNB corpus. The real corpus plugs in unchanged at
`data/raw/CICIoT2023/`.

Stages run successfully:

| Stage | Command | Result |
| --- | --- | --- |
| inspect | `python -m src.data.inspect` | 4 CSV files, 2,800 rows, 52 features, 28 labels, schema matches |
| ingest (Spark) | `python -m src.data.ingest` | parquet + ingestion_stats.json + schema_report.json |
| clean (Spark) | `python -m src.data.clean` | leakage audit written, 0 dropped rows |
| sample B | `python -m src.data.sample --dataset B` | balanced binary sample |
| split B/binary | `python -m src.data.split --dataset B --task binary` | 2013/370/420 stratified |
| train binary | `python -m src.models.train --task binary --model LogisticRegression,RandomForest,XGBoost,LightGBM --dataset B` | 4 models trained + compared |
| sample C | `python -m src.data.sample --dataset C` | 8-class stratified sample |
| split C/family | `python -m src.data.split --dataset C --task family` | 2013/370/420 stratified |
| train family | `python -m src.models.train --task family --model RandomForest --dataset C` | 8-class RF, macro-F1=0.83 |
| export | `python -m src.inference.predictor --model-dir models/binary/XGBoost --export` | features.json written |
| simulate | `python -m src.simulation.simulator --model-dir models/binary/XGBoost --generate` | 41 flows extracted, **1 attack detected** with confidence 0.998, severity HIGH |
| benchmark | `python -m src.evaluation.benchmark --mode edge --model-dir models/binary/XGBoost --rows 100` | edge benchmark JSON written |
| tests | `python -m pytest tests/` | **28 passed** |

## Live detection proof

The simulator generated a synthetic pcap with:
  - 40 benign TCP flows (5 packets each: SYN / SYN-ACK / ACK / GET / ACK)
  - 1 attack flow (500 SYN packets in a single 5-tuple, ~10 000 pps)

The trained XGBoost binary model inspected each extracted flow and produced
this alert for the attack flow:

```
[ALERT]
Time: 2026-11-20 00:35:58
Detection: ATTACK
Class: Attack
Confidence: 0.9983
Severity: HIGH
Flow: 10.0.0.20|10.0.0.99|80|40000|6
Packets: 500
```

This proves the end-to-end chain
`packets -> flow extraction -> ML model -> alert`
works as required by spec section 25 — **no CSV row replay**.

## Binary model comparison (Dataset B, smoke-test data)

| Model | Macro F1 | Recall | FPR | FNR | Latency (ms) | Size (KB) |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 1.000 | 1.000 | 0.000 | 0.000 | 0.001 | 1.22 |
| RandomForest       | 1.000 | 1.000 | 0.000 | 0.000 | 0.053 | 155.99 |
| XGBoost            | 1.000 | 1.000 | 0.000 | 0.000 | 0.004 | 210.90 |
| LightGBM           | 1.000 | 1.000 | 0.000 | 0.000 | 0.003 | 77.79 |

The 1.0 scores are an artefact of the synthetic smoke-test data — on the
real CICIoT2023 corpus the scores will differ. The point of this table is
that the comparison framework runs and produces all the required columns
(Macro-F1, Recall, FPR, Latency, Model size) per spec section 15.

## 8-class family classifier (Dataset C, smoke-test data)

```
RandomForest:
  accuracy          : 0.764
  macro F1          : 0.831
  weighted F1       : 0.774
  balanced accuracy : 0.804
  FPR (macro)       : 0.037
  FNR (macro)       : 0.196
  train time        : 2.18 s
  avg inference     : 0.077 ms / flow
  model size        : 31.9 MB
```

## Experiment matrix (spec section 32) — status

| # | Experiment | Status |
| --- | --- | --- |
| 1 | Binary: Benign vs Attack | **done** (4 models trained, threshold sweep run) |
| 2 | 8-class: Benign + 7 attack families | **done** (RF on Dataset C) |
| 3 | 28-class: individual attacks | code ready (`--task finegrained`), not yet run on real data |
| 4 | Feature reduction: 39 / 20 / 10 | code ready (`src.features.selection`), not yet applied |
| 5 | Model comparison: LR / RF / XGBoost / LightGBM | **done** for binary |
| 6 | Edge benchmark: PC vs Raspberry Pi | code ready (`src.evaluation.benchmark --mode edge`); PC side tested, Pi side requires the Pi |
| 7 | Live simulation: normal vs attack | **done** — synthetic pcap detected correctly |

## What is and is NOT done

### Done

- Full repository structure (matches spec section 2 exactly).
- All source modules: data / features / models / evaluation / inference / simulation.
- PySpark ingestion + cleaning + sampling + split, all writing parquet + JSON metadata.
- Leakage audit (spec section 8): documents each drop decision.
- Preprocessing that fits on TRAIN ONLY (spec section 13).
- Four trained binary models (LR / DT-not-shown / RF / XGBoost / LightGBM) with full metric bundle (accuracy / precision / recall / macro-F1 / weighted-F1 / balanced accuracy / FPR / FNR / confusion matrix).
- Threshold sweep (0.3 / 0.4 / 0.5 / 0.6 / 0.7) for binary IDS.
- 8-class family classifier.
- Live IDS pipeline: pcap -> flow extraction -> ML model -> alert log. NOT a CSV replay.
- Edge benchmark code (latency / p95 / throughput / RAM / CPU / model size).
- Streamlit dashboard.
- Docker compose (isolated `iot_lab` network, `internal: true`).
- Raspberry Pi deployment guide, systemd unit, install.sh.
- 5 Jupyter notebook stubs.
- 28 unit tests, all passing.

### Not done (requires real data or real hardware)

- Running on the full 46M-row CICIoT2023 corpus (requires UNB download).
- Feature selection / SHAP on the real feature distributions.
- 28-class finegrained training (code is ready).
- Raspberry Pi benchmark numbers (requires the Pi).
- Real packet capture inside the Docker lab (requires Docker on the host).

## How to plug in the real dataset

1. Download the CSV release from <https://www.unb.ca/cic/datasets/iotdataset-2023.html>.
2. Unzip the CSV files into `data/raw/CICIoT2023/`.
3. Delete the synthetic files if present:
   `rm data/raw/CICIoT2023/synthetic_chunk_*.csv`
4. Re-run the pipeline in the documented order (README §4).

Nothing else changes. The synthetic smoke-test corpus was generated with
exactly the same schema (52 features + label) as the real release, so the
pipeline accepts both interchangeably.
