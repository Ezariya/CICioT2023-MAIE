"""
src/features/schema.py — feature schema helpers.

Loads the expected feature list from config/features.yaml and exposes:
  - FEATURE_COLUMNS      (ordered list of feature names)
  - LABEL_COLUMN         (target column name)
  - TREE_MODELS          (models that do NOT need scaling)
  - LINEAR_MODELS        (models that DO need scaling)
"""
from __future__ import annotations

from typing import List

from src.utils import get_feature_columns, get_label_column, load_config

FEATURE_COLUMNS: List[str] = get_feature_columns()
LABEL_COLUMN: str = get_label_column()

# Tree-based models are scale-invariant
TREE_MODELS = {"RandomForest", "DecisionTree", "XGBoost", "LightGBM"}
LINEAR_MODELS = {"LogisticRegression"}


def features_for(model_name: str) -> List[str]:
    """Return the feature list. (Same list for every model — scaling differs.)"""
    return list(FEATURE_COLUMNS)


def needs_scaling(model_name: str) -> bool:
    """Return True iff the model requires feature scaling."""
    cfg = load_config()
    if cfg["preprocessing"]["scale_for_tree_models"]:
        return True
    return model_name in LINEAR_MODELS


def expected_label_values() -> List[str]:
    """Return every label value the dataset is allowed to contain."""
    from src.utils import all_attack_labels
    return ["Benign"] + all_attack_labels()


if __name__ == "__main__":
    print(f"feature count: {len(FEATURE_COLUMNS)}")
    print(f"label column : {LABEL_COLUMN}")
    print(f"first 5      : {FEATURE_COLUMNS[:5]}")
