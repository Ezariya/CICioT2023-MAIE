"""
src/dashboard/app.py — Streamlit dashboard for the live IDS (spec section 30).

Run with:
    streamlit run src/dashboard/app.py

Panels:
  - Total / Benign / Attack flows (cumulative)
  - Attack distribution (bar)
  - Current detection + confidence
  - Recent alerts
  - CPU / RAM

The dashboard polls simulation/traffic/alerts.log and reads model metadata,
so it does not retrain or interfere with detection.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

import streamlit as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils import load_config, project_root


def _read_alert_log(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    alerts: List[Dict] = []
    current: Dict | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("---"):
            if current:
                alerts.append(current)
                current = None
            continue
        if current is None:
            current = {}
        if ":" in line:
            k, v = line.split(":", 1)
            current[k.strip()] = v.strip()
    if current:
        alerts.append(current)
    return alerts


def main():
    cfg = load_config()
    st.set_page_config(page_title="CICIoT2023 IDS", layout="wide")
    st.title("CICIoT2023 AI-based IoT IDS — Live Dashboard")

    default_model_dir = project_root() / "models" / "binary"
    model_dir = default_model_dir
    if model_dir.exists():
        for sub in sorted(model_dir.iterdir()):
            if (sub / "metadata.json").exists():
                model_dir = sub
                break

    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        st.sidebar.markdown(f"**Model**: {meta.get('model', '?')}")
        st.sidebar.markdown(f"**Task**: {meta.get('task', '?')}")
        st.sidebar.markdown(f"**Features**: {meta.get('feature_count', '?')}")
        st.sidebar.markdown(f"**Threshold**: {meta.get('threshold', '?')}")
        if "metrics" in meta:
            m = meta["metrics"]
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"Macro F1 : `{m.get('f1_macro', '?')}`")
            st.sidebar.markdown(f"Recall   : `{m.get('recall_macro', '?')}`")
            st.sidebar.markdown(f"FPR      : `{m.get('fpr', '?')}`")
            st.sidebar.markdown(f"FNR      : `{m.get('fnr', '?')}`")

    alerts_log = project_root() / cfg["simulation"]["alert"]["log_path"]
    alerts = _read_alert_log(alerts_log)

    attack_alerts = [a for a in alerts if a.get("Detection") == "ATTACK"]
    benign_seen = sum(1 for a in alerts if a.get("Detection") == "BENIGN")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total flows", len(alerts))
    col2.metric("Benign", benign_seen)
    col3.metric("Attacks", len(attack_alerts))
    col4.metric("Alert rate",
                f"{100 * len(attack_alerts) / max(len(alerts), 1):.1f}%")

    if attack_alerts:
        cls_counter = Counter(a.get("Class", "?") for a in attack_alerts)
        df = pd.DataFrame(cls_counter.items(), columns=["Attack", "Count"])
        st.subheader("Attack distribution")
        st.bar_chart(df.set_index("Attack"))

    st.subheader("Latest detection")
    if alerts:
        latest = alerts[-1]
        st.json(latest)
    else:
        st.info("No traffic captured yet. Run the simulator first.")

    st.subheader("Recent alerts (max 50)")
    if attack_alerts:
        df_alerts = pd.DataFrame(attack_alerts[-50:])
        st.dataframe(df_alerts, use_container_width=True)
    else:
        st.write("No attack alerts yet.")

    try:
        import psutil
        proc = psutil.Process(os.getpid())
        col_a, col_b = st.columns(2)
        col_a.metric("Dashboard RAM (MB)", round(proc.memory_info().rss / 1024 / 1024, 1))
        col_b.metric("System RAM used (%)", psutil.virtual_memory().percent)
    except ImportError:
        st.caption("psutil not installed — CPU/RAM panel disabled")

    st.caption("Refresh every %d s (configure in config.yaml:dashboard.refresh_sec)"
               % cfg["dashboard"]["refresh_sec"])


if __name__ == "__main__":
    main()
