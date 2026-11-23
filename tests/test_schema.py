"""
tests/test_schema.py — unit tests for schema and label taxonomy.

Run with: pytest tests/ -v
"""
import pytest

from src.utils import (
    all_attack_labels,
    attack_to_family,
    family_to_index,
    is_benign,
    to_binary_label,
    to_family_label,
)
from src.features.schema import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    LINEAR_MODELS,
    TREE_MODELS,
    expected_label_values,
    needs_scaling,
)


def test_feature_count_ish():
    """We expect roughly 47 flow features. Adjust the lower bound if you add
    new derived features."""
    assert len(FEATURE_COLUMNS) >= 40, "feature schema looks incomplete"


def test_label_column():
    assert LABEL_COLUMN == "label"


def test_attack_to_family_known():
    assert attack_to_family("DDoS-SYN_Flood") == "DDoS"
    assert attack_to_family("Mirai-udpplain_flood") == "Mirai"
    assert attack_to_family("MITM-ArpSpoofing") == "Spoofing"
    assert attack_to_family("DictionaryBruteForce") == "Brute Force"


def test_attack_to_family_unknown_raises():
    with pytest.raises(ValueError):
        attack_to_family("not-a-real-attack")


def test_binary_label():
    assert to_binary_label("Benign") == 0
    assert to_binary_label("DDoS-SYN_Flood") == 1
    assert is_benign("Benign")
    assert not is_benign("DDoS-SYN_Flood")


def test_family_label():
    assert to_family_label("Benign") == "Benign"
    assert to_family_label("DDoS-SYN_Flood") == "DDoS"
    assert to_family_label("SqlInjection") == "Web-based"


def test_family_to_index_eight_classes():
    idx = family_to_index()
    assert len(idx) == 8
    assert idx["Benign"] == 0


def test_all_attack_labels_count():
    """The CICIoT2023 dataset documents 27 individual attack types in 7 families
    (DDoS 6 + DoS 4 + Recon 4 + Mirai 4 + Spoofing 2 + Web-based 6 + BruteForce 1)."""
    labels = all_attack_labels()
    assert len(labels) == 27


def test_expected_label_values_includes_benign_and_attacks():
    vals = expected_label_values()
    assert "Benign" in vals
    assert "DDoS-SYN_Flood" in vals
    assert len(vals) == 28   # 1 benign + 27 attacks


def test_needs_scaling():
    assert needs_scaling("LogisticRegression") is True
    assert needs_scaling("RandomForest") is False
    assert needs_scaling("XGBoost") is False
    assert needs_scaling("LightGBM") is False
    assert needs_scaling("DecisionTree") is False
