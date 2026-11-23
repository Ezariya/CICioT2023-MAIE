"""
tests/test_predictor.py — test model loader + predictor on a tiny synthetic model.
"""
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.features.preprocess import FlowPreprocessor
from src.features.schema import FEATURE_COLUMNS
from src.inference.model_loader import load_model_dir
from src.inference.predictor import predict_one


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Train a tiny RF and write model.joblib + preprocessing + metadata.json."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, (200, len(FEATURE_COLUMNS))),
                     columns=FEATURE_COLUMNS)
    y = (X.iloc[:, 0] > 0).astype(int).values
    rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    rf.fit(X.values, y)

    pre = FlowPreprocessor(feature_columns=FEATURE_COLUMNS, scale=False)
    pre.fit(X)

    d = tmp_path / "RF"
    d.mkdir()
    joblib.dump(rf, d / "model.joblib")
    joblib.dump(pre, d / "preprocessing.joblib")
    (d / "features.json").write_text(json.dumps({"features": FEATURE_COLUMNS}))
    (d / "metadata.json").write_text(json.dumps({
        "model": "RandomForest",
        "task": "binary",
        "version": "test",
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "training_dataset": "test",
        "seed": 42,
        "metrics": {},
        "threshold": 0.5,
        "needs_scaling": False,
    }))
    return d


def test_load_model_dir(tmp_model_dir):
    loaded = load_model_dir(tmp_model_dir)
    assert loaded.model is not None
    assert loaded.preprocessor is not None
    assert loaded.task == "binary"
    assert loaded.threshold == 0.5
    assert len(loaded.features) == len(FEATURE_COLUMNS)


def test_predict_one_returns_expected_keys(tmp_model_dir):
    loaded = load_model_dir(tmp_model_dir)
    row = {f: 0.0 for f in FEATURE_COLUMNS}
    out = predict_one(loaded, row)
    for k in ["prediction", "label", "confidence", "inference_ms"]:
        assert k in out


def test_predict_one_high_first_feature_predicts_attack(tmp_model_dir):
    """The toy RF was trained so feature 0 > 0 => label 1."""
    loaded = load_model_dir(tmp_model_dir)
    row = {f: 0.0 for f in FEATURE_COLUMNS}
    row[FEATURE_COLUMNS[0]] = 5.0   # clearly > 0
    out = predict_one(loaded, row)
    assert out["prediction"] == 1
    assert out["label"] == "Attack"


def test_predict_one_low_first_feature_predicts_benign(tmp_model_dir):
    loaded = load_model_dir(tmp_model_dir)
    row = {f: 0.0 for f in FEATURE_COLUMNS}
    row[FEATURE_COLUMNS[0]] = -5.0
    out = predict_one(loaded, row)
    assert out["prediction"] == 0
    assert out["label"] == "Benign"
