"""
src/evaluation/metrics.py — IDS metrics that go beyond accuracy.

Hard rule (spec section 10): NEVER use accuracy_score alone. Always report:
  precision, recall, F1, macro-F1, weighted-F1, balanced accuracy,
  confusion matrix, FPR, FNR.

For IDS the priority is:
  - Recall on attacks
  - False Negative Rate (FNR) — missed attacks
  - False Positive Rate (FPR) — false alarms
  - Macro-F1
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_full_metrics(y_true, y_pred, task: str = "binary") -> Dict[str, float]:
    """Compute the full metric bundle for a classification task."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true).tolist()))
    metrics: Dict[str, float] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro":    round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro":        round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_weighted":     round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "confusion_matrix": cm.tolist(),
    }

    if task == "binary":
        # cm = [[TN, FP], [FN, TP]]
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["tn"] = int(tn); metrics["fp"] = int(fp)
            metrics["fn"] = int(fn); metrics["tp"] = int(tp)
            metrics["fpr"] = round(fp / max(fp + tn, 1), 6)   # FP / (FP + TN)
            metrics["fnr"] = round(fn / max(fn + tp, 1), 6)   # FN / (FN + TP)
            metrics["tpr"] = round(tp / max(tp + fn, 1), 6)   # recall on attacks
            metrics["tnr"] = round(tn / max(tn + fp, 1), 6)   # recall on benign
            # precision / recall for the ATTACK class (label 1)
            metrics["precision_attack"] = round(tp / max(tp + fp, 1), 6)
            metrics["recall_attack"]    = round(tp / max(tp + fn, 1), 6)
        else:
            # single-class edge case
            metrics["fpr"] = 0.0; metrics["fnr"] = 0.0
    else:
        # multiclass: report macro FPR / FNR (one-vs-rest averaged)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        fpr = fp / np.maximum(fp + tn, 1)
        fnr = fn / np.maximum(fn + tp, 1)
        metrics["fpr_macro"] = round(float(np.mean(fpr)), 6)
        metrics["fnr_macro"] = round(float(np.mean(fnr)), 6)
        metrics["fpr_per_class"] = [round(float(x), 6) for x in fpr]
        metrics["fnr_per_class"] = [round(float(x), 6) for x in fnr]
    return metrics


def threshold_sweep(y_true, proba_attack, thresholds: List[float]) -> Dict[float, Dict[str, float]]:
    """Sweep thresholds for binary IDS (spec section 18)."""
    out = {}
    y_true = np.asarray(y_true)
    proba_attack = np.asarray(proba_attack)
    for t in thresholds:
        y_pred = (proba_attack >= t).astype(int)
        m = compute_full_metrics(y_true, y_pred, task="binary")
        out[f"{t:.2f}"] = {
            "f1_macro": m["f1_macro"],
            "recall_attack": m.get("recall_attack", 0.0),
            "precision_attack": m.get("precision_attack", 0.0),
            "fpr": m.get("fpr", 0.0),
            "fnr": m.get("fnr", 0.0),
        }
    return out


def false_negative_analysis(y_true, y_pred, labels: List[str], attack_classes: List[str]) -> Dict:
    """
    Spec section 19: which attacks are easiest to miss?

    Returns per-class FN rate, sorted descending.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fnr_per_class = {}
    for i, cls in enumerate(labels):
        fn = cm[i].sum() - cm[i, i]
        tp = cm[i, i]
        total = fn + tp
        fnr = fn / total if total > 0 else 0.0
        if cls in attack_classes:
            fnr_per_class[cls] = {
                "fn": int(fn), "tp": int(tp), "total": int(total),
                "fnr": round(fnr, 6),
            }
    # sort by FNR descending — these are the attacks we miss the most
    sorted_by_fnr = dict(sorted(fnr_per_class.items(), key=lambda kv: -kv[1]["fnr"]))
    return {"false_negative_analysis": sorted_by_fnr}
