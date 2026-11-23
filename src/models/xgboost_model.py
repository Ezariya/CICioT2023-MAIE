"""
src/models/xgboost_model.py — XGBoost / LightGBM wrappers.

Both are gradient-boosted tree models. Tree-based, scale-invariant, and
typically the strongest single-table classifiers. LightGBM is included as a
fallback because XGBoost may not be installable on every environment.
"""
from __future__ import annotations

from typing import Any, Dict


def build_xgboost(params: Dict[str, Any] | None = None, random_state: int = 42):
    """Build an XGBoost classifier. Falls back to LightGBM if XGBoost missing."""
    p = dict(params or {})
    p.setdefault("n_estimators", 300)
    p.setdefault("max_depth", 10)
    p.setdefault("learning_rate", 0.1)
    p.setdefault("subsample", 0.9)
    p.setdefault("colsample_bytree", 0.9)
    p.setdefault("tree_method", "hist")
    p.setdefault("n_jobs", -1)
    p.setdefault("random_state", random_state)
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise RuntimeError(
            "xgboost is not installed. Install with: pip install xgboost"
        ) from e
    return XGBClassifier(**p)


def build_lightgbm(params: Dict[str, Any] | None = None, random_state: int = 42):
    """Build a LightGBM classifier."""
    p = dict(params or {})
    p.setdefault("n_estimators", 400)
    p.setdefault("max_depth", -1)
    p.setdefault("learning_rate", 0.1)
    p.setdefault("num_leaves", 64)
    p.setdefault("class_weight", "balanced")
    p.setdefault("n_jobs", -1)
    p.setdefault("random_state", random_state)
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise RuntimeError(
            "lightgbm is not installed. Install with: pip install lightgbm"
        ) from e
    return LGBMClassifier(**p)
