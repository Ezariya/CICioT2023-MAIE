"""
src/evaluation/benchmark.py — model + edge benchmarking.

Two levels of evaluation (spec section 31):
  ML level       : accuracy / precision / recall / F1 / FPR / FNR / CM
  System level   : latency / throughput / RAM / CPU / model size / delay

This module measures the SYSTEM level:
  - average inference time (ms / row)
  - p95 inference time
  - throughput (flows / sec)
  - CPU / RAM (psutil if available)
  - model size on disk

Modes:
  --mode model   : benchmark one model on a given dataset
  --mode edge    : same but tagged as edge (Raspberry Pi or PC).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.utils import project_root, load_config, write_json
from src.features.preprocess import FlowPreprocessor


def _try_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def measure_inference(model, X: np.ndarray, n_warmup: int = 10) -> Dict[str, float]:
    """Run inference on X and return timing stats."""
    timings_ms: List[float] = []
    # warmup
    for i in range(min(n_warmup, len(X))):
        model.predict(X[i:i + 1])
    # real run, per-row
    for i in range(len(X)):
        t0 = time.perf_counter()
        model.predict(X[i:i + 1])
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(timings_ms)
    return {
        "avg_inference_ms":  round(float(arr.mean()), 4),
        "p50_inference_ms":  round(float(np.percentile(arr, 50)), 4),
        "p95_inference_ms":  round(float(np.percentile(arr, 95)), 4),
        "p99_inference_ms":  round(float(np.percentile(arr, 99)), 4),
        "max_inference_ms":  round(float(arr.max()), 4),
        "min_inference_ms":  round(float(arr.min()), 4),
        "throughput_flows_per_sec": round(1000.0 / float(arr.mean()), 2) if arr.mean() > 0 else 0.0,
        "n_rows": int(len(arr)),
    }


def system_snapshot() -> Dict[str, float]:
    """Capture RAM / CPU usage."""
    out: Dict[str, float] = {"ram_mb": 0.0, "cpu_percent": 0.0}
    p = _try_psutil()
    if p is None:
        return out
    proc = p.Process(os.getpid())
    out["ram_mb"] = round(proc.memory_info().rss / 1024 / 1024, 2)
    out["cpu_percent"] = round(proc.cpu_percent(interval=0.5), 2)
    out["system_ram_total_mb"] = round(p.virtual_memory().total / 1024 / 1024, 2)
    out["system_ram_used_pct"] = round(p.virtual_memory().percent, 2)
    return out


def benchmark_model_dir(model_dir: Path, X: pd.DataFrame, n_rows: int | None = None) -> Dict:
    """Load model + preprocessing from a dir and benchmark."""
    model = joblib.load(model_dir / "model.joblib")
    pre = joblib.load(model_dir / "preprocessing.joblib")
    meta_path = model_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    feature_cols = meta.get("features") or list(X.columns)
    if n_rows is not None:
        X = X.iloc[:n_rows]
    X_proc = pre.transform(X[feature_cols])
    snap_before = system_snapshot()
    timing = measure_inference(model, X_proc)
    snap_after = system_snapshot()
    size_kb = (model_dir / "model.joblib").stat().st_size / 1024
    return {
        "model": meta.get("model", "?"),
        "task": meta.get("task", "?"),
        "feature_count": len(feature_cols),
        "model_size_kb": round(size_kb, 2),
        "n_rows": timing["n_rows"],
        "timing": timing,
        "ram_mb": snap_after["ram_mb"],
        "cpu_percent": snap_after["cpu_percent"],
        "host": _detect_host(),
        "metrics_from_metadata": meta.get("metrics", {}),
    }


def _detect_host() -> str:
    """Best-effort: detect if we're running on a Raspberry Pi."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            txt = f.read()
        if "Raspberry Pi" in txt or "BCM" in txt:
            return "raspberry_pi"
    except Exception:
        pass
    return "pc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["model", "edge"])
    ap.add_argument("--model-dir", required=True,
                    help="path to model directory containing model.joblib")
    ap.add_argument("--parquet", required=False,
                    help="optional test parquet to sample rows from")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    if args.parquet:
        # Use Spark to load, then convert small slice to pandas
        from src.data.ingest import get_spark
        spark = get_spark()
        df = spark.read.parquet(args.parquet).limit(args.rows).toPandas()
        spark.stop()
        # drop the label column
        from src.features.schema import FEATURE_COLUMNS, LABEL_COLUMN
        X = df[[c for c in FEATURE_COLUMNS if c in df.columns]]
    else:
        # synthesize random feature rows
        from src.features.schema import FEATURE_COLUMNS
        X = pd.DataFrame(
            np.random.rand(args.rows, len(FEATURE_COLUMNS)),
            columns=FEATURE_COLUMNS,
        )

    res = benchmark_model_dir(Path(args.model_dir), X, n_rows=args.rows)
    out_path = args.out or (
        project_root() / "reports" / f"benchmark_{args.mode}_{res['model']}.json"
    )
    write_json(res, out_path)
    print(json.dumps(res, indent=2, default=str))
    print(f"[benchmark] wrote {out_path}")


if __name__ == "__main__":
    main()
