"""
CICIoT2023 IDS — Streamlit Demo Dashboard
==========================================

A visual proof-of-concept dashboard for your trained IDS model.

Run:
    streamlit run demo_dashboard.py -- --model-dir models/binary/XGBoost

Or if no model-dir is given, it auto-detects the first available model.

Shows:
  - Model info (accuracy, F1, FPR, FNR) in the sidebar
  - "Run Benign Scenario" button — generates 8 HTTP flows
  - "Run Attack Scenario" button — generates a SYN flood
  - "Run Mixed Scenario" button — both at once
  - Real-time alert feed
  - Flow table with predictions
  - Live stats (total flows, attacks, detection rate, throughput)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

import streamlit as st

# Add repo to path
REPO = Path(__file__).resolve().parent
if (REPO / "src").exists():
    sys.path.insert(0, str(REPO))

# Suppress scapy verbosity
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CICIoT2023 IDS Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers — reuse the logic from demo.py
# ---------------------------------------------------------------------------
def make_packet(src, dst, sport, dport, flags, ts, payload=None):
    from scapy.all import IP, TCP, Ether, Raw
    if payload:
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags) / Raw(load=payload)
    else:
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    p.time = ts
    return p


def gen_benign(base, n_flows=8, **kwargs):
    n = kwargs.get("n", n_flows)
    pkts = []
    for f in range(n):
        sport = 50000 + f
        t = base + f * 0.3
        pkts.append(make_packet("10.0.0.10", "10.0.0.20", sport, 80, "S", t))
        pkts.append(make_packet("10.0.0.20", "10.0.0.10", 80, sport, "SA", t + 0.01))
        pkts.append(make_packet("10.0.0.10", "10.0.0.20", sport, 80, "A", t + 0.02))
        pkts.append(make_packet("10.0.0.10", "10.0.0.20", sport, 80, "PA", t + 0.03,
                                payload=f"GET /sensor/{f} HTTP/1.1\r\nHost: iot.local\r\n\r\n"))
        pkts.append(make_packet("10.0.0.20", "10.0.0.10", 80, sport, "A", t + 0.04))
    return pkts


def gen_attack(base, n=300):
    pkts = []
    for i in range(n):
        pkts.append(make_packet("10.0.0.99", "10.0.0.20", 40000, 80, "S",
                                base + i * 0.0001))
    return pkts


def gen_portscan(base, n=100):
    pkts = []
    for i in range(n):
        pkts.append(make_packet("10.0.0.99", "10.0.0.20", 40000, 1 + i, "S",
                                base + i * 0.01))
    return pkts


def extract_and_predict(loaded, packets, scenario):
    from scapy.all import wrpcap
    from src.simulation.flow_extraction import pcap_to_flow_rows
    from src.inference.predictor import predict_one

    tmp = Path(tempfile.mktemp(suffix=".pcap"))
    wrpcap(str(tmp), packets)
    df = pcap_to_flow_rows(tmp)
    tmp.unlink()

    results = []
    for idx, row in df.iterrows():
        feat = {c: float(row.get(c, 0.0)) for c in loaded.features}
        pred = predict_one(loaded, feat)
        results.append({
            "Time": time.strftime("%H:%M:%S"),
            "Scenario": scenario,
            "Flow": str(row.get("_flow_key", f"flow_{idx}")),
            "Packets": int(row.get("_packet_count", 0)),
            "Prediction": "ATTACK" if pred.get("prediction") == 1 else "BENIGN",
            "Label": pred.get("label", ""),
            "Confidence": round(pred.get("confidence", 0), 4),
            "Inference_ms": round(pred.get("inference_ms", 0), 3),
            "Severity": ("HIGH" if (pred.get("confidence", 0) or 0) > 0.85 else
                         "MEDIUM" if (pred.get("confidence", 0) or 0) > 0.65 else
                         "LOW") if pred.get("prediction") == 1 else "—",
        })
    return results


# ---------------------------------------------------------------------------
# Auto-find model(s)
# ---------------------------------------------------------------------------
def find_models():
    found = []
    for root in [Path("."), Path(".."), Path("/content/ciciot2023-ids"), REPO]:
        for subdir in ["binary", "multiclass"]:
            base = root / "models" / subdir
            if not base.exists():
                continue
            for name in ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression", "DecisionTree"]:
                p = base / name
                if (p / "model.joblib").exists():
                    p_str = str(p)
                    if p_str not in found:
                        found.append(p_str)
    return found


# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_dir: str):
    from src.inference.model_loader import load_model_dir
    return load_model_dir(model_dir)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ CICIoT2023 IDS")
st.sidebar.markdown("---")

available_models = find_models()
if available_models:
    model_dir = st.sidebar.selectbox("Select Model", available_models, index=0)
    loaded = load_model(model_dir)
    meta = loaded.metadata
    algo_name = meta.get('model') or Path(model_dir).name
    st.sidebar.markdown(f"**Algorithm**: {algo_name}")
    st.sidebar.markdown(f"**Task**: {loaded.task}")
    st.sidebar.markdown(f"**Features**: {len(loaded.features)}")
    st.sidebar.markdown(f"**Threshold**: {loaded.threshold}")
    m = meta.get("metrics", {})
    if m:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Training metrics**")
        col1, col2 = st.sidebar.columns(2)
        f1_val = m.get('f1_macro', m.get('f1', 0))
        acc_val = m.get('accuracy', 0)
        fpr_val = m.get('fpr', m.get('FPR', 0))
        fnr_val = m.get('fnr', m.get('FNR', 0))
        col1.metric("Macro-F1", f"{float(f1_val):.3f}")
        col2.metric("Accuracy", f"{float(acc_val):.3f}")
        col1.metric("FPR", f"{float(fpr_val):.3f}")
        col2.metric("FNR", f"{float(fnr_val):.3f}")
else:
    st.sidebar.error("No model found!\nTrain a model first or specify path.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Scenarios**")
run_benign = st.sidebar.button("🌐 Run Benign Traffic", use_container_width=True)
run_attack = st.sidebar.button("⚠️ Run SYN Flood Attack", use_container_width=True)
run_mixed = st.sidebar.button("🔀 Run Mixed Scenario", use_container_width=True)
run_scan = st.sidebar.button("🔍 Run Port Scan", use_container_width=True)
clear = st.sidebar.button("🗑️ Clear Results", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("CICIoT2023 AI-Based IoT IDS")
st.markdown("### Live Detection Demo")
st.markdown("""
This dashboard demonstrates that your trained model can detect cyber attacks
in IoT network traffic. Click a scenario button in the sidebar to generate
synthetic traffic, extract flow features, and run real-time inference.
""")

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = []

if clear:
    st.session_state.results = []
    st.rerun()

# Run scenarios
base_time = time.time()
if run_benign:
    with st.spinner("Generating benign HTTP traffic..."):
        pkts = gen_benign(base_time, n_flows=8)
        results = extract_and_predict(loaded, pkts, "Benign")
        st.session_state.results.extend(results)
    st.success(f"Generated {len(pkts)} packets -> {len(results)} flows analyzed")

if run_attack:
    with st.spinner("Generating SYN flood attack (300 packets)..."):
        pkts = gen_attack(base_time + 1, n=300)
        results = extract_and_predict(loaded, pkts, "SYN Flood")
        st.session_state.results.extend(results)
    st.success(f"Generated {len(pkts)} attack packets -> {len(results)} flows analyzed")

if run_mixed:
    with st.spinner("Generating mixed traffic..."):
        b = gen_benign(base_time + 2, n_flows=5)
        a = gen_attack(base_time + 2.5, n=200)
        results = extract_and_predict(loaded, b + a, "Mixed")
        st.session_state.results.extend(results)
    st.success(f"Generated {len(b)} benign + {len(a)} attack -> {len(results)} flows")

if run_scan:
    with st.spinner("Generating port scan (100 ports)..."):
        pkts = gen_portscan(base_time + 3, n=100)
        results = extract_and_predict(loaded, pkts, "Port Scan")
        st.session_state.results.extend(results)
    st.success(f"Generated {len(pkts)} scan packets -> {len(results)} flows")

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
results = st.session_state.results
total = len(results)
attacks = sum(1 for r in results if r["Prediction"] == "ATTACK")
benign = total - attacks
avg_inf = np.mean([r["Inference_ms"] for r in results]) if results else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Flows", total)
col2.metric("Benign", benign)
col3.metric("Attacks", attacks)
col4.metric("Detection Rate", f"{100*attacks/max(total,1):.0f}%")
col5.metric("Throughput", f"{1000/max(avg_inf,0.001):,.0f} fps")

st.markdown("---")

# ---------------------------------------------------------------------------
# Two-column layout: alerts + flow table
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 🚨 Recent Alerts")
    attack_results = [r for r in results if r["Prediction"] == "ATTACK"]
    if attack_results:
        for r in attack_results[-10:]:  # last 10 alerts
            severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟠"}.get(r["Severity"], "⚪")
            st.markdown(f"""
            <div style="background: rgba(255,0,0,0.08); border-left: 3px solid red;
                        padding: 10px; margin: 5px 0; border-radius: 4px;">
                <b>{severity_color} {r['Severity']} — {r['Label']}</b><br/>
                <small style="color: gray;">{r['Time']} | Flow: {r['Flow']}</small><br/>
                <small>Confidence: <b>{r['Confidence']:.4f}</b> | Packets: {r['Packets']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No attacks detected yet. Run an attack scenario.")

with col_right:
    st.markdown("### 📊 Flow Detection Table")
    if results:
        df = pd.DataFrame(results)
        # Color-code predictions
        st.dataframe(
            df[["Time", "Scenario", "Flow", "Packets", "Prediction",
                "Label", "Confidence", "Severity", "Inference_ms"]],
            use_container_width=True,
            height=400,
        )
    else:
        st.info("Click a scenario button in the sidebar to start.")

# ---------------------------------------------------------------------------
# Attack distribution chart
# ---------------------------------------------------------------------------
if attacks > 0:
    st.markdown("---")
    st.markdown("### Attack Distribution")
    attack_labels = Counter(r["Label"] for r in attack_results)
    chart_df = pd.DataFrame(attack_labels.items(), columns=["Attack Type", "Count"])
    st.bar_chart(chart_df.set_index("Attack Type"))

# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### How It Works")
st.markdown("""
```
  Synthetic Traffic (scapy)
         │
         ▼
  PCAP file (temporary)
         │
         ▼
  Flow Extraction (src/simulation/flow_extraction.py)
  ┌──────────────────────────────────────┐
  │ packets → 5-tuple grouping → stats   │
  │ → 52 CICIoT2023-compatible features  │
  └──────────────────────────────────────┘
         │
         ▼
  ML Inference (src/inference/predictor.py)
  ┌──────────────────────────────────────┐
  │ preprocessing.joblib → model.joblib  │
  │ → prediction + confidence            │
  └──────────────────────────────────────┘
         │
         ▼
  Alert (severity: HIGH / MEDIUM / LOW)
```
""")

# Footer
st.markdown("---")
st.caption("CICIoT2023 AI-Based IoT IDS — Proof of Concept Demo")
