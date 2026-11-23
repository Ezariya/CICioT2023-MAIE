"""
FIXED train.py for Colab — clean LabelEncoder handling for multiclass.

This file REPLACES src/models/train.py entirely. It fixes:
  1. UnboundLocalError on `metrics` (caused by broken regex edits)
  2. "Mix of label input types (string and number)" for multiclass
     (caused by inconsistent LabelEncoder application)

Key change: for family/finegrained tasks, ALL label partitions
(train/val/test) are encoded with the SAME LabelEncoder BEFORE model.fit().
Predictions are therefore integer-encoded, matching y_test.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from src.utils import (
    ensure_dir,
    family_to_index,
    load_config,
    load_feature_schema,
    project_root,
    set_global_seed,
    write_json,
)
from src.features.schema import FEATURE_COLUMNS, LABEL_COLUMN, needs_scaling
from src.features.preprocess import FlowPreprocessor, build_preprocessor_for
from src.models.baseline import build_baseline
from src.models.random_forest import build_random_forest
from src.models.xgboost_model import build_xgboost, build_lightgbm
from src.evaluation.metrics import compute_full_metrics, threshold_sweep
from src.evaluation.confusion import plot_confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data loading (TRAIN / VAL / TEST parquet, written by src.data.split)
# ---------------------------------------------------------------------------
def _spark():
    from src.data.ingest import get_spark
    return get_spark()


def load_split_pandas(dataset_letter: str, task: str):
    """Load train/val/test parquet via Spark -> pandas (allowed: small samples)."""
    sample_dir = (
        project_root() / "data" / "processed"
        / {"A": "binary", "B": "binary", "C": "multiclass", "D": "edge"}[dataset_letter.upper()]
        / f"sampled_{dataset_letter.lower()}.parquet"
    )
    split_dir = sample_dir.parent / f"{task}_split"
    spark = _spark()
    out = {}
    for name in ["train", "val", "test"]:
        p = split_dir / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"split parquet missing: {p}. Run src.data.split --dataset {dataset_letter} --task {task} first."
            )
        out[name] = spark.read.parquet(str(p)).toPandas()
    spark.stop()
    return out["train"], out["val"], out["test"]


def xy_split(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["y"]
    return X, y


# ---------------------------------------------------------------------------
# Imbalance handling
# ---------------------------------------------------------------------------
def apply_imbalance_strategy(X: pd.DataFrame, y: pd.Series, strategy: str, task: str):
    if strategy in ("none", "class_weight"):
        return X, y, strategy
    if strategy == "undersampling":
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_r, y_r = rus.fit_resample(X, y)
        return X_r, y_r, strategy
    if strategy == "oversampling":
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_r, y_r = ros.fit_resample(X, y)
        return X_r, y_r, strategy
    raise ValueError(f"unknown imbalance strategy: {strategy}")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(model_name: str, cfg: Dict[str, Any], task: str):
    seed = cfg["project"]["seed"]
    params = None
    for entry in cfg["models"]["baselines"] + cfg["models"]["candidates"]:
        if entry["name"] == model_name:
            params = entry.get("params", {})
            break
    if model_name == "LogisticRegression":
        return build_baseline("LogisticRegression", params, random_state=seed)
    if model_name == "DecisionTree":
        return build_baseline("DecisionTree", params, random_state=seed)
    if model_name == "RandomForest":
        return build_random_forest(params, random_state=seed)
    if model_name == "XGBoost":
        m = build_xgboost(params, random_state=seed)
        if task == "binary":
            m.set_params(objective="binary:logistic", eval_metric="logloss")
        return m
    if model_name == "LightGBM":
        return build_lightgbm(params, random_state=seed)
    raise ValueError(f"unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Train one (model, task, dataset)  —  CLEAN IMPLEMENTATION
# ---------------------------------------------------------------------------
def train_one(
    model_name: str,
    task: str,
    dataset_letter: str,
    imbalance_strategy: str = "class_weight",
    feature_cols: List[str] | None = None,
) -> Dict[str, Any]:
    cfg = load_config()
    set_global_seed(cfg["project"]["seed"])

    if feature_cols is None:
        feature_cols = list(FEATURE_COLUMNS)

    # --- 1. Load splits ---------------------------------------------------
    train_df, val_df, test_df = load_split_pandas(dataset_letter, task)
    X_train, y_train = xy_split(train_df, feature_cols)
    X_val,   y_val   = xy_split(val_df,   feature_cols)
    X_test,  y_test  = xy_split(test_df,  feature_cols)

    # --- 2. Encode string labels for multiclass (CLEAN FIX) ---------------
    # For binary, y is already 0/1 integer (set by split.py's _add_task_label).
    # For family / finegrained, y is a string — XGBoost/LightGBM need integers.
    # We fit ONE LabelEncoder on the union of all labels and transform ALL
    # three partitions consistently. Predictions are then integer, matching
    # the encoded y_test — no type mismatch in metrics.
    label_encoder = None
    label_classes = None
    if task != "binary" and y_train.dtype == "object":
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        all_labels = list(set(
            y_train.values.tolist()
            + y_val.values.tolist()
            + y_test.values.tolist()
        ))
        label_encoder.fit(all_labels)
        label_classes = list(label_encoder.classes_)
        y_train = pd.Series(
            label_encoder.transform(y_train.values),
            index=y_train.index, name="y",
        )
        y_val = pd.Series(
            label_encoder.transform(y_val.values),
            index=y_val.index, name="y",
        )
        y_test = pd.Series(
            label_encoder.transform(y_test.values),
            index=y_test.index, name="y",
        )

    # --- 3. Imbalance (only modifies TRAIN) -------------------------------
    X_train_res, y_train_res, used_strategy = apply_imbalance_strategy(
        X_train, y_train, imbalance_strategy, task
    )

    # --- 4. Preprocessing: FIT ON TRAIN ONLY ------------------------------
    pre = build_preprocessor_for(model_name, feature_cols)
    pre.fit(X_train_res)
    Xtr = pre.transform(X_train_res)
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)

    # --- 5. Model ---------------------------------------------------------
    model = build_model(model_name, cfg, task)
    t0 = time.time()
    model.fit(Xtr, y_train_res.values)
    train_time_s = time.time() - t0

    # --- 6. Predictions ---------------------------------------------------
    t1 = time.time()
    if task == "binary" and hasattr(model, "predict_proba"):
        proba_test = model.predict_proba(Xte)[:, 1]
        y_pred = (proba_test >= 0.5).astype(int)
        latency_ms = (time.time() - t1) * 1000 / max(len(y_pred), 1)
        sweep = threshold_sweep(y_test.values, proba_test, cfg["thresholds"]["binary"])
    else:
        y_pred = model.predict(Xte)
        proba_test = None
        latency_ms = (time.time() - t1) * 1000 / max(len(y_pred), 1)
        sweep = None

    # --- 7. Metrics (both y_test and y_pred are same type now) -----------
    metrics = compute_full_metrics(y_test.values, y_pred, task=task)
    metrics["train_time_s"] = round(train_time_s, 3)
    metrics["avg_inference_ms"] = round(latency_ms, 4)
    if sweep:
        metrics["threshold_sweep"] = sweep
    if label_classes is not None:
        metrics["label_classes"] = label_classes

    # --- 8. Model size ----------------------------------------------------
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as fp:
        joblib.dump(model, fp.name)
        model_size_kb = os.path.getsize(fp.name) / 1024
        os.unlink(fp.name)
    metrics["model_size_kb"] = round(model_size_kb, 2)

    # --- 9. Persist artefacts --------------------------------------------
    task_subdir = {"binary": "binary", "family": "multiclass", "finegrained": "multiclass"}[task]
    model_dir = project_root() / "models" / task_subdir / model_name
    ensure_dir(model_dir)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(pre,   model_dir / "preprocessing.joblib")
    if label_encoder is not None:
        joblib.dump(label_encoder, model_dir / "label_encoder.joblib")

    # Write features.json
    (model_dir / "features.json").write_text(
        json.dumps({"features": feature_cols}, indent=2)
    )

    metadata = {
        "model": model_name,
        "task": task,
        "version": cfg["project"]["version"],
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "training_dataset": dataset_letter.upper(),
        "imbalance_strategy": used_strategy,
        "seed": cfg["project"]["seed"],
        "metrics": metrics,
        "threshold": 0.5,
        "needs_scaling": needs_scaling(model_name),
        "label_classes": label_classes,
    }
    write_json(metadata, model_dir / "metadata.json")
    write_json(metadata, project_root() / "reports" / f"metrics_{task}_{model_name}.json")

    # --- 10. Confusion matrix --------------------------------------------
    try:
        labels_sorted = sorted(np.unique(y_test).tolist())
        display_labels = labels_sorted
        if label_encoder is not None:
            display_labels = [
                label_encoder.classes_[i] if i < len(label_encoder.classes_) else str(i)
                for i in labels_sorted
            ]
        plot_confusion_matrix(
            y_test.values, y_pred, labels=labels_sorted,
            out=project_root() / "reports" / f"cm_{task}_{model_name}.png",
            title=f"{model_name} ({task})",
        )
    except Exception as e:
        print(f"[train] confusion plot skipped: {e}")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["binary", "family", "finegrained"])
    ap.add_argument("--model", required=True,
                    help="comma-separated list: LogisticRegression,DecisionTree,RandomForest,XGBoost,LightGBM")
    ap.add_argument("--dataset", default="B",
                    help="dataset letter A/B/C/D (default B for binary)")
    ap.add_argument("--imbalance", default="class_weight",
                    choices=["none", "class_weight", "undersampling", "oversampling"])
    args = ap.parse_args()

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    print(f"[train] task={args.task} dataset={args.dataset} strategy={args.imbalance}")
    print(f"[train] models={models}")
    rows = []
    for m in models:
        meta = train_one(m, args.task, args.dataset, args.imbalance)
        print(f"\n=== {m} ({args.task}) ===")
        for k, v in meta["metrics"].items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"     {kk}: {vv}")
            elif isinstance(v, list) and len(v) > 6:
                print(f"  {k}: [{v[0]}, {v[1]}, ... ({len(v)} items)]")
            else:
                print(f"  {k}: {v}")
        rows.append({
            "model": m,
            "macro_f1": meta["metrics"].get("f1_macro"),
            "recall_macro": meta["metrics"].get("recall_macro"),
            "fpr": meta["metrics"].get("fpr") or meta["metrics"].get("fpr_macro"),
            "fnr": meta["metrics"].get("fnr") or meta["metrics"].get("fnr_macro"),
            "latency_ms": meta["metrics"].get("avg_inference_ms"),
            "model_size_kb": meta["metrics"].get("model_size_kb"),
        })

    comparison = pd.DataFrame(rows)
    out_csv = project_root() / "reports" / f"comparison_{args.task}_{args.dataset}.csv"
    comparison.to_csv(out_csv, index=False)
    print(f"\n[comparison]\n{comparison.to_string(index=False)}")
    print(f"[comparison] wrote {out_csv}")


if __name__ == "__main__":
    main()
