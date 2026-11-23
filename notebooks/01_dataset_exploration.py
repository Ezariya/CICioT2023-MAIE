# notebooks/01_dataset_exploration.ipynb (rendered as Python stub)
#
# Open in Jupyter:  jupyter lab notebooks/01_dataset_exploration.ipynb
# This is the .py representation. To create the .ipynb version, run:
#   jupytext --to ipynb notebooks/01_dataset_exploration.py
#
# Goal (spec section 7):
#   - dataset size (files, rows, features)
#   - label distribution (33 attacks grouped into 7 families + Benign)
#   - per-feature dtype / min / max / mean / std / missing % / infinite %
#   - highly correlated / constant / near-constant features
#   - extreme values
#
# All computation uses PySpark (NOT pandas on the full corpus).

import json
from pathlib import Path

import yaml

from src.utils import load_config, load_feature_schema, project_root

cfg = load_config()
schema = load_feature_schema()
meta_path = project_root() / "data" / "metadata" / "ingestion_stats.json"
print(f"Loading ingestion stats from {meta_path}")
meta = json.loads(meta_path.read_text())

print(f"\n=== Dataset size ===")
print(f"Files   : {meta['number_of_files']}")
print(f"Rows    : {meta['number_of_rows']:,}")
print(f"Features: {meta['number_of_features']}")
print(f"Labels  : {len(meta['unique_labels'])}")

print(f"\n=== Label distribution (33 attacks -> 7 families + Benign) ===")
for lab, n in sorted(meta["class_counts"].items(), key=lambda kv: -kv[1]):
    print(f"   {lab:<30} {n:>12,}")

print(f"\n=== Per-feature statistics ===")
import pandas as pd
fs = meta["feature_stats"]
df_stats = pd.DataFrame([f for f in fs if f.get("present")])
print(df_stats[["name", "dtype", "missing_pct", "infinite_pct", "min", "max", "mean", "std"]].to_string(index=False))

# Highly correlated features — compute on a small pandas sample later
# (Use PySpark correlation matrix on a sample for the full dataset.)
