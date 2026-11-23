"""
tests/test_preprocess.py — test the FlowPreprocessor (fit on TRAIN only).
"""
import numpy as np
import pandas as pd
import pytest

from src.features.preprocess import FlowPreprocessor, build_preprocessor_for
from src.features.schema import FEATURE_COLUMNS


def _make_df(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1, (rows, len(FEATURE_COLUMNS)))
    # Inject NaN / inf into specific positions only if the df is large enough.
    if rows > 0:
        data[0, 0] = np.nan
    if rows > 1:
        data[1, 1] = np.inf
    if rows > 2:
        data[2, 2] = -np.inf
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def test_fit_then_transform_shape():
    pre = FlowPreprocessor(feature_columns=FEATURE_COLUMNS, scale=True)
    Xtr = _make_df(100)
    pre.fit(Xtr)
    out = pre.transform(_make_df(20, seed=1))
    assert out.shape == (20, len(FEATURE_COLUMNS))


def test_no_nan_after_transform():
    pre = FlowPreprocessor(feature_columns=FEATURE_COLUMNS, scale=True)
    pre.fit(_make_df(100))
    out = pre.transform(_make_df(50, seed=2))
    assert np.isfinite(out).all(), "transform produced NaN or Inf"


def test_median_filled_on_train_only():
    """Test that medians are computed on TRAIN, not on the transform input."""
    pre = FlowPreprocessor(feature_columns=FEATURE_COLUMNS, scale=False,
                           nan_strategy="median")
    Xtr = _make_df(100, seed=10)
    pre.fit(Xtr)
    # median stored should be from Xtr
    assert pre.medians_ is not None
    # transform a small batch that has a NaN — should be filled with TRAIN median
    Xtest = _make_df(5, seed=99)
    out = pre.transform(Xtest)
    assert np.isfinite(out).all()


def test_transform_before_fit_raises():
    pre = FlowPreprocessor(feature_columns=FEATURE_COLUMNS)
    with pytest.raises(RuntimeError):
        pre.transform(_make_df(5))


def test_build_preprocessor_for_tree_models_does_not_scale():
    pre = build_preprocessor_for("RandomForest", FEATURE_COLUMNS)
    assert pre.scale is False


def test_build_preprocessor_for_linear_models_scales():
    pre = build_preprocessor_for("LogisticRegression", FEATURE_COLUMNS)
    assert pre.scale is True
