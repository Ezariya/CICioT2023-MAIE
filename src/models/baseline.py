"""
src/models/baseline.py — simple baselines (spec section 14).

  Baseline 1: Logistic Regression
  Baseline 2: Decision Tree
  Baseline 3: Random Forest (also a baseline candidate)

These are intentionally simple. They establish a floor before we test
XGBoost / LightGBM.
"""
from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_baseline(name: str, params: Dict[str, Any] | None = None, random_state: int = 42):
    """Instantiate a baseline model from config.yaml params."""
    params = dict(params or {})
    if name == "LogisticRegression":
        params.setdefault("max_iter", 1000)
        params.setdefault("class_weight", "balanced")
        params.setdefault("n_jobs", -1)
        params.setdefault("random_state", random_state)
        return LogisticRegression(**params)
    if name == "DecisionTree":
        params.setdefault("max_depth", 20)
        params.setdefault("class_weight", "balanced")
        params.setdefault("random_state", random_state)
        return DecisionTreeClassifier(**params)
    if name == "RandomForest":
        params.setdefault("n_estimators", 200)
        params.setdefault("max_depth", 30)
        params.setdefault("class_features", "sqrt")
        params.setdefault("class_weight", "balanced_subsample")
        params.setdefault("n_jobs", -1)
        params.setdefault("random_state", random_state)
        return RandomForestClassifier(**params)
    raise ValueError(f"unknown baseline: {name}")
