"""
src/inference/predictor.py — single-row / batch inference.

Loads a model directory with model_loader, transforms a flow-feature row,
and returns a prediction + confidence.

For binary IDS the prediction is 0/1 with probability of attack, and the
threshold is read from metadata.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.inference.model_loader import LoadedModel, load_model_dir


def predict_one(loaded: LoadedModel, row: Dict) -> Dict:
    """Predict a single flow row (dict of feature_name -> value)."""
    df = pd.DataFrame([row], columns=loaded.features)
    if loaded.preprocessor is not None:
        try:
            X = loaded.preprocessor.transform(df)
        except Exception:
            X = df.fillna(0.0).values
    else:
        X = df.fillna(0.0).values
    t0 = time.perf_counter()
    if hasattr(loaded.model, "predict_proba"):
        proba = loaded.model.predict_proba(X)[0]
        if loaded.task == "binary":
            p_attack = float(proba[1]) if len(proba) > 1 else float(proba[0])
            pred = int(p_attack >= loaded.threshold)
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "prediction": pred,
                "label": "Attack" if pred == 1 else "Benign",
                "confidence": round(p_attack if pred == 1 else (1 - p_attack), 6),
                "attack_probability": round(p_attack, 6),
                "threshold": loaded.threshold,
                "inference_ms": round(latency_ms, 4),
            }
        else:
            # multiclass
            idx = int(np.argmax(proba))
            classes = list(getattr(loaded.model, "classes_", range(len(proba))))
            label = classes[idx]
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "prediction": idx,
                "label": str(label),
                "confidence": round(float(proba[idx]), 6),
                "class_probabilities": [round(float(p), 6) for p in proba],
                "inference_ms": round(latency_ms, 4),
            }
    else:
        pred = int(loaded.model.predict(X)[0])
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "prediction": pred,
            "label": str(pred),
            "confidence": None,
            "inference_ms": round(latency_ms, 4),
        }


def predict_batch(loaded: LoadedModel, rows: List[Dict]) -> List[Dict]:
    return [predict_one(loaded, r) for r in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--export", action="store_true",
                    help="write features.json next to model.joblib then exit")
    ap.add_argument("--row", action="append",
                    help="feature=value, repeatable; if omitted, runs a smoke predict on random row")
    args = ap.parse_args()

    loaded = load_model_dir(args.model_dir)
    if args.export:
        from src.inference.model_loader import write_features_json
        write_features_json(args.model_dir, loaded.features)
        print(f"[predictor] wrote features.json in {args.model_dir}")
        return

    if args.row:
        row = {}
        for kv in args.row:
            k, v = kv.split("=", 1)
            try:
                v = float(v)
            except ValueError:
                pass
            row[k] = v
    else:
        # smoke predict: random row
        np.random.seed(42)
        row = {f: float(np.random.rand()) for f in loaded.features}
        print("[predictor] no --row given, predicting a random smoke row")

    result = predict_one(loaded, row)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
