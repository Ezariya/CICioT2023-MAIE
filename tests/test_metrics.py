"""
tests/test_metrics.py — test that metrics go beyond accuracy.
"""
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_full_metrics,
    false_negative_analysis,
    threshold_sweep,
)


def test_binary_metrics_keys():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0, 0, 1, 1, 1, 0])
    m = compute_full_metrics(y, p, task="binary")
    for k in [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "f1_weighted", "balanced_accuracy", "confusion_matrix",
        "fpr", "fnr", "tpr", "tnr", "precision_attack", "recall_attack",
    ]:
        assert k in m, f"missing key {k}"


def test_perfect_prediction_has_zero_fpr_fnr():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    m = compute_full_metrics(y, p, task="binary")
    assert m["fpr"] == 0.0
    assert m["fnr"] == 0.0
    assert m["f1_macro"] == 1.0


def test_all_wrong_prediction():
    y = np.array([0, 0, 1, 1])
    p = np.array([1, 1, 0, 0])
    m = compute_full_metrics(y, p, task="binary")
    assert m["fpr"] == 1.0
    assert m["fnr"] == 1.0
    assert m["f1_macro"] == 0.0


def test_threshold_sweep_keys():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    sweep = threshold_sweep(y, proba, [0.3, 0.5, 0.7])
    for t in ["0.30", "0.50", "0.70"]:
        assert t in sweep
        for k in ["f1_macro", "recall_attack", "precision_attack", "fpr", "fnr"]:
            assert k in sweep[t]


def test_false_negative_analysis():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 0, 2, 0])
    res = false_negative_analysis(
        y, p, labels=[0, 1, 2], attack_classes=[1, 2],
    )
    fna = res["false_negative_analysis"]
    # class 1: 1 TP, 1 FN -> FNR 0.5
    # class 2: 1 TP, 1 FN -> FNR 0.5
    assert "1" in fna or 1 in fna
