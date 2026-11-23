"""
src/features/selection.py — feature importance + feature reduction.

Methods (spec section 16):
  - random_forest_importance   (fast, tree-based)
  - permutation_importance     (model-agnostic, slower)
  - shap                       (optional, on a small sample)

Then test reductions: top 30 / 20 / 15 / 10 features.

The objective is NOT maximising accuracy. We want the minimum useful feature
set with acceptable detection performance — this is what makes the model
deployable on a Raspberry Pi.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.utils import load_config, project_root, write_json


def random_forest_importance(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, seed: int = 42
) -> pd.Series:
    """Fit a quick RF and return Gini importances sorted descending."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=15, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed,
    )
    rf.fit(X_train.values, y_train.values)
    imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    return imp.sort_values(ascending=False)


def permutation_importance_for(
    fitted_model, X_val: pd.DataFrame, y_val: pd.Series, scoring: str = "f1_macro",
    n_repeats: int = 5, seed: int = 42, n_jobs: int = -1,
) -> pd.Series:
    """Compute permutation importance on a validation set."""
    r = permutation_importance(
        fitted_model, X_val.values, y_val.values,
        scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs,
    )
    imp = pd.Series(r.importances_mean, index=X_val.columns)
    return imp.sort_values(ascending=False)


def shap_importance(fitted_model, X_sample: pd.DataFrame) -> pd.Series:
    """Approximate SHAP importance on a small sample (optional)."""
    try:
        import shap
    except ImportError as e:
        raise RuntimeError("shap is not installed; install with: pip install shap") from e
    # TreeExplainer works for RF / XGBoost / LightGBM
    explainer = shap.TreeExplainer(fitted_model)
    shap_values = explainer.shap_values(X_sample.values)
    # Multi-class -> list of arrays; binary -> array of shape (n_samples, n_features)
    if isinstance(shap_values, list):
        arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        arr = np.abs(shap_values)
    if arr.ndim == 3:  # (samples, features, classes)
        arr = arr.mean(axis=2)
    importance = arr.mean(axis=0)
    return pd.Series(importance, index=X_sample.columns).sort_values(ascending=False)


def select_top_k(importance: pd.Series, k: int) -> List[str]:
    """Return the top-k feature names from an importance series."""
    return list(importance.head(k).index)


def evaluate_top_k(
    importance: pd.Series, ks: List[int]
) -> Dict[int, List[str]]:
    """Return {k: [feature names]} for each k in ks."""
    return {k: select_top_k(importance, k) for k in ks}


def main():
    """CLI: read an existing model's feature importance from reports/ and
    write top-k feature lists to data/processed/feature_sets.json."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance-file", default="reports/feature_importance.json")
    ap.add_argument("--top-k", default="39,30,20,15,10")
    args = ap.parse_args()
    path = project_root() / args.importance_file
    if not path.exists():
        raise FileNotFoundError(
            f"importance file not found: {path}. Train a model first to compute it."
        )
    with path.open("r") as f:
        imp_data = json.load(f)
    imp = pd.Series(imp_data["importances"], index=imp_data["features"]).sort_values(ascending=False)
    ks = [int(k) for k in args.top_k.split(",")]
    sets = evaluate_top_k(imp, ks)
    out_path = project_root() / "data" / "processed" / "feature_sets.json"
    write_json(
        {"method": imp_data.get("method", "?"), "feature_sets": sets},
        out_path,
    )
    print(f"[selection] wrote {out_path}")
    for k, names in sets.items():
        print(f"  top-{k}: {names}")


if __name__ == "__main__":
    main()
