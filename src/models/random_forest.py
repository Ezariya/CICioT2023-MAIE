"""
src/models/random_forest.py — Random Forest wrapper.

RF is expected to be a strong candidate because the data is tabular and
flow-feature based.
"""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(params: Dict[str, Any] | None = None, random_state: int = 42):
    p = dict(params or {})
    p.setdefault("n_estimators", 200)
    p.setdefault("max_depth", 30)
    p.setdefault("max_features", "sqrt")
    p.setdefault("class_weight", "balanced_subsample")
    p.setdefault("n_jobs", -1)
    p.setdefault("random_state", random_state)
    return RandomForestClassifier(**p)
