"""
src/features/preprocess.py — one reproducible preprocessing pipeline.

  Raw features
      │
      ▼
  NaN / Inf handling         (median fill, ±1e15 clip — fit on TRAIN)
      │
      ▼
  Feature validation         (drop leaked cols, dtype check)
      │
      ▼
  Selected features          (subset from selection.py, optional)
      │
      ▼
  Scaling if required        (StandardScaler — fit on TRAIN)
      │
      ▼
  ML model

Hard rule (spec section 13): fit preprocessing ONLY on TRAIN. Never on the
full corpus before splitting. This class therefore exposes .fit(X_train)
and .transform(X_other) explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, project_root, write_json


class FlowPreprocessor(BaseEstimator, TransformerMixin):
    """Stateful preprocessing for CICIoT2023 flow features.

    Parameters
    ----------
    feature_columns : list[str]
        Ordered feature column names.
    scale : bool
        If True, fit a StandardScaler on TRAIN data.
    nan_strategy : str
        "median"  -> fill NaN with column median (fit on TRAIN)
        "zero"    -> fill NaN with 0
    inf_clip : float
        Replace |x| > inf_clip with ±inf_clip BEFORE computing the median.
    """

    def __init__(
        self,
        feature_columns: List[str],
        scale: bool = True,
        nan_strategy: str = "median",
        inf_clip: float = 1e15,
    ):
        self.feature_columns = list(feature_columns)
        self.scale = scale
        self.nan_strategy = nan_strategy
        self.inf_clip = inf_clip
        # fitted state
        self.medians_: Optional[pd.Series] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_in_: List[str] = list(self.feature_columns)

    # ------------------------------------------------------------------
    def _clip_inf(self, X: pd.DataFrame) -> pd.DataFrame:
        # Replace +-inf with NaN first, then we will fill NaN.
        X = X.replace([np.inf, -np.inf], np.nan)
        # Also clip absurdly large finite values
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].clip(-self.inf_clip, self.inf_clip)
        return X

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        """Fit medians + scaler on TRAIN data only."""
        # Reorder + cast
        X = X.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        X = self._clip_inf(X)
        if self.nan_strategy == "median":
            self.medians_ = X.median(axis=0, skipna=True)
        elif self.nan_strategy == "zero":
            self.medians_ = pd.Series(0.0, index=self.feature_columns)
        else:
            raise ValueError(f"unknown nan_strategy: {self.nan_strategy}")
        # fill (so scaler sees finite data)
        X_filled = X.fillna(self.medians_)
        if self.scale:
            self.scaler_ = StandardScaler().fit(X_filled.values)
        else:
            self.scaler_ = None
        # record that fit happened
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("FlowPreprocessor.transform called before .fit()")
        X = X.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        X = self._clip_inf(X)
        X = X.fillna(self.medians_)
        if self.scaler_ is not None:
            return self.scaler_.transform(X.values)
        return X.values

    # ------------------------------------------------------------------
    def to_metadata(self) -> dict:
        """Return a JSON-serialisable description (no NumPy arrays)."""
        return {
            "feature_columns": list(self.feature_columns),
            "scale": self.scale,
            "nan_strategy": self.nan_strategy,
            "inf_clip": self.inf_clip,
            "medians": None if self.medians_ is None else self.medians_.to_dict(),
            "scaler_mean": None if self.scaler_ is None else list(self.scaler_.mean_),
            "scaler_scale": None if self.scaler_ is None else list(self.scaler_.scale_),
        }


def build_preprocessor_for(model_name: str, feature_columns: List[str]) -> FlowPreprocessor:
    """Return a preprocessor configured for the given model (scaling on/off)."""
    cfg = load_config()
    if model_name == "any":
        scale = True
    elif model_name in {"LogisticRegression"}:
        scale = cfg["preprocessing"]["scale_for_linear_models"]
    else:
        scale = cfg["preprocessing"]["scale_for_tree_models"]
    return FlowPreprocessor(
        feature_columns=feature_columns,
        scale=scale,
        nan_strategy=cfg["preprocessing"]["nan_strategy"],
        inf_clip=1e15,
    )
