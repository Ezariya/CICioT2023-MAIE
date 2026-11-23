"""
src/evaluation/confusion.py — confusion-matrix plotting.

Renders with matplotlib + Noto Sans SC so Chinese characters and Unicode
render correctly. Saves PNGs to reports/.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def plot_confusion_matrix(
    y_true, y_pred, labels: List, out: Path, title: str = ""
):
    """Render a confusion matrix PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from matplotlib import font_manager as fm
        fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
    except Exception:
        pass
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.6)),
                          constrained_layout=True)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)),
           yticks=range(len(labels)),
           xticklabels=labels,
           yticklabels=labels,
           ylabel="True label",
           xlabel="Predicted label",
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # numeric annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
