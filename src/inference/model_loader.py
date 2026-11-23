"""
src/inference/model_loader.py — load an exported model + preprocessing.

Hard rule (spec section 20): the inference system loads model.joblib +
preprocessing.joblib + features.json + metadata.json. It NEVER retrains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np


@dataclass
class LoadedModel:
    model: object
    preprocessor: object
    features: List[str]
    metadata: dict
    threshold: float
    model_dir: Path

    @property
    def task(self) -> str:
        return self.metadata.get("task", "binary")

    @property
    def needs_proba(self) -> bool:
        return self.task == "binary"


def load_model_dir(model_dir: str | Path) -> LoadedModel:
    """Load all artefacts from a model directory."""
    p = Path(model_dir)
    model = joblib.load(p / "model.joblib")
    pre = joblib.load(p / "preprocessing.joblib")
    meta = json.loads((p / "metadata.json").read_text())
    features = meta.get("features") or []
    threshold = meta.get("threshold", 0.5)
    if not features:
        # fallback to features.json if metadata lacks features
        fj = p / "features.json"
        if fj.exists():
            features = json.loads(fj.read_text())["features"]
    return LoadedModel(
        model=model,
        preprocessor=pre,
        features=features,
        metadata=meta,
        threshold=threshold,
        model_dir=p,
    )


def write_features_json(model_dir: str | Path, features: List[str]):
    """Convenience: write features.json next to model.joblib."""
    p = Path(model_dir)
    (p / "features.json").write_text(
        json.dumps({"features": list(features)}, indent=2)
    )
