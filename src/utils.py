"""
CICIoT2023 AI-based IoT IDS — shared utilities.

Centralised config / feature-schema loader used by every module so the
project has one source of truth. Also provides reproducible seeding.
"""
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Project root = parent of this file's parent (src/utils.py -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """Return the absolute path to the repository root."""
    return PROJECT_ROOT


def load_config() -> Dict[str, Any]:
    """Load and cache the global config.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_schema() -> Dict[str, Any]:
    """Load features.yaml (feature list, leakage rules, validation)."""
    path = PROJECT_ROOT / "config" / "features.yaml"
    if not path.exists():
        raise FileNotFoundError(f"feature schema not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names from features.yaml."""
    schema = load_feature_schema()
    return [f["name"] for f in schema["features"]]


def get_label_column() -> str:
    """Return the label column name."""
    return load_feature_schema()["target"]["name"]


def set_global_seed(seed: int = 42) -> None:
    """Seed Python / NumPy / PySpark RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Spark RNG is seeded per-call in the relevant modules.


def ensure_dir(path: os.PathLike | str) -> Path:
    """Create directory if it does not exist and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: os.PathLike | str, indent: int = 2) -> Path:
    """Write a JSON file with UTF-8 and indentation."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)
    return p


def read_json(path: os.PathLike | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Label taxonomy helpers
# ---------------------------------------------------------------------------
_ATTACK_TO_FAMILY_CACHE: Dict[str, str] | None = None


def _build_attack_to_family() -> Dict[str, str]:
    """Map every individual attack label to its 7-family (or Benign) bucket."""
    global _ATTACK_TO_FAMILY_CACHE
    if _ATTACK_TO_FAMILY_CACHE is not None:
        return _ATTACK_TO_FAMILY_CACHE
    cfg = load_config()
    mapping: Dict[str, str] = {}
    for family_key, family_info in cfg["label_taxonomy"].items():
        family_name = family_info["family"]
        for atk in family_info["attacks"]:
            mapping[atk] = family_name
    _ATTACK_TO_FAMILY_CACHE = mapping
    return mapping


def attack_to_family(label: str) -> str:
    """Map an individual attack label (e.g. DDoS-SYN_Flood) to its family."""
    m = _build_attack_to_family()
    if label not in m:
        raise ValueError(f"unknown label '{label}' — update config.yaml label_taxonomy")
    return m[label]


def is_benign(label: str) -> bool:
    return label.strip().lower() == "benign"


def to_binary_label(label: str) -> int:
    """0 = Benign, 1 = Attack."""
    return 0 if is_benign(label) else 1


def to_family_label(label: str) -> str:
    """Map label to one of 8 classes (Benign + 7 attack families)."""
    if is_benign(label):
        return "Benign"
    return attack_to_family(label)


def family_to_index() -> Dict[str, int]:
    cfg = load_config()
    return {name: i for i, name in enumerate(cfg["tasks"]["family"]["classes"])}


def all_attack_labels() -> list[str]:
    """Return every individual attack label (used by the finegrained task)."""
    cfg = load_config()
    out: list[str] = []
    for fam_key, fam_info in cfg["label_taxonomy"].items():
        for atk in fam_info["attacks"]:
            if atk != "Benign":
                out.append(atk)
    return out
