"""
CICIoT2023 AI-Powered IoT Intrusion Detection System — SOC Dashboard
=====================================================================

A presentation-grade, interactive Security Operations Center (SOC) dashboard
for evaluating, simulating, and validating AI/ML models on IoT network traffic.

Features:
  1. Live Traffic Simulation: Scapy packet generator -> Flow extraction -> ML inference
  2. Benchmark Dataset Replay: Stream real CICIoT2023 dataset flows (34 attack types + benign)
  3. PCAP File Inspector: Upload any .pcap/.pcapng file to extract and classify flows
  4. Model Analytics & Explainability: Feature importances, latency benchmarks, metrics
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup repository path
REPO = Path(__file__).resolve().parent
if (REPO / "src").exists():
    sys.path.insert(0, str(REPO))

# Suppress scapy runtime logs
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
try:
    from scapy.all import conf, wrpcap
    conf.verb = 0
except ImportError:
    pass

from src.simulation.flow_extraction import pcap_to_flow_rows
from src.simulation.traffic_generator import (
    gen_benign_http,
    gen_benign_mqtt,
    gen_syn_flood,
    gen_udp_flood,
    gen_port_scan,
    gen_mirai_botnet,
    gen_mixed_fleet,
)
from src.inference.model_loader import load_model_dir
from src.inference.predictor import predict_one

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CICIoT2023 AI IDS | SOC Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark-mode modern SOC aesthetic
st.markdown("""
<style>
    .metric-card {
        background: #1e222d;
        border: 1px solid #2d3343;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 18px;
        font-weight: 600;
    }
    .alert-high {
        background: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .alert-medium {
        background: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .alert-benign {
        background: rgba(16, 185, 129, 0.12);
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Model Discovery & Loading
# ---------------------------------------------------------------------------
def discover_models() -> List[str]:
    """Find all trained model directories in models/."""
    found = []
    roots = [Path("."), Path(".."), REPO, Path("/content/ciciot2023-ids")]
    for root in roots:
        for subdir in ["binary", "multiclass"]:
            base = root / "models" / subdir
            if not base.exists():
                continue
            for name in ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression", "family_XGBoost", "finegrained_XGBoost"]:
                p = base / name
                if (p / "model.joblib").exists():
                    p_str = str(p.resolve())
                    if p_str not in found:
                        found.append(p_str)
            # check remaining subdirectories
            for p in sorted(base.iterdir()):
                if p.is_dir() and (p / "model.joblib").exists():
                    p_str = str(p.resolve())
                    if p_str not in found:
                        found.append(p_str)
    return found


@st.cache_resource
def get_model(model_dir: str):
    return load_model_dir(model_dir)


@st.cache_data
def get_benchmark_samples() -> Optional[pd.DataFrame]:
    """Load benchmark samples from lightweight demo sample or full parquet."""
    candidates = [
        REPO / "data" / "demo_benchmark_sample.parquet",
        Path("data/demo_benchmark_sample.parquet"),
        REPO / "data" / "processed" / "sampled_ciciot2023.parquet",
        Path("data/processed/sampled_ciciot2023.parquet"),
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if len(df) > 5000:
                    df = df.sample(3000, random_state=42)
                return df
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Helper: Predict PCAP Packets
# ---------------------------------------------------------------------------
def process_packets(loaded, packets: List, scenario_title: str) -> List[Dict]:
    """Extract flows from scapy packets and run inference."""
    tmp = Path(tempfile.mktemp(suffix=".pcap"))
    wrpcap(str(tmp), packets)
    df = pcap_to_flow_rows(tmp)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

    results = []
    for idx, row in df.iterrows():
        feat = {c: float(row.get(c, 0.0)) for c in loaded.features}
        pred = predict_one(loaded, feat)
        is_attack = (pred.get("prediction") == 1)
        conf = float(pred.get("confidence", 0.0) or 0.0)
        severity = ("HIGH" if conf > 0.85 else "MEDIUM" if conf > 0.65 else "LOW") if is_attack else "BENIGN"

        results.append({
            "Time": time.strftime("%H:%M:%S"),
            "Scenario": scenario_title,
            "Flow Key": str(row.get("_flow_key", f"flow_{idx}")),
            "Packets": int(row.get("_packet_count", 0)),
            "Verdict": "ATTACK" if is_attack else "BENIGN",
            "Class": pred.get("label", "Attack" if is_attack else "Benign"),
            "Confidence": round(conf, 4),
            "Latency (ms)": round(float(pred.get("inference_ms", 0.0)), 2),
            "Severity": severity,
            "raw_features": feat,
        })
    return results


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []

# ---------------------------------------------------------------------------
# Sidebar: Model Config & Controls
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("CICIoT2023 IDS")
st.sidebar.caption("AI-Powered Network Threat Detection")
st.sidebar.markdown("---")

available_models = discover_models()
if not available_models:
    st.sidebar.error("No model directories found in models/. Please verify model files.")
    st.stop()

# Format model display names
model_names = {}
for p in available_models:
    path_obj = Path(p)
    parent = path_obj.parent.name
    name = path_obj.name
    display = f"{name} ({parent})"
    model_names[display] = p

chosen_display = st.sidebar.selectbox("Active AI Model", list(model_names.keys()), index=0)
active_model_dir = model_names[chosen_display]
loaded = get_model(active_model_dir)
meta = loaded.metadata

st.sidebar.markdown(f"**Task**: `{loaded.task.upper()}`")
st.sidebar.markdown(f"**Features**: `{len(loaded.features)} inputs`")
st.sidebar.markdown(f"**Threshold**: `{loaded.threshold}`")

m = meta.get("metrics", {})
if m:
    with st.sidebar.expander("Training Benchmarks", expanded=True):
        col1, col2 = st.columns(2)
        f1_v = m.get("f1_macro", m.get("f1", 0.0))
        acc_v = m.get("accuracy", 0.0)
        fpr_v = m.get("fpr", m.get("FPR", 0.0))
        fnr_v = m.get("fnr", m.get("FNR", 0.0))
        col1.metric("Macro-F1", f"{float(f1_v):.3f}")
        col2.metric("Accuracy", f"{float(acc_v):.3f}")
        col1.metric("FPR", f"{float(fpr_v):.3f}")
        col2.metric("FNR", f"{float(fnr_v):.3f}")

st.sidebar.markdown("---")

# Quick Actions
if st.sidebar.button("🗑️ Clear Live Feed", use_container_width=True):
    st.session_state.results = []
    st.rerun()

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🛡️ IoT Intrusion Detection System")
    st.markdown(
        f"Real-time evaluation and cyber-defense platform powered by **{Path(active_model_dir).name}** "
        f"trained on the **CICIoT2023** benchmark."
    )
with col_h2:
    total_analyzed = len(st.session_state.results)
    attacks_detected = sum(1 for r in st.session_state.results if r["Verdict"] == "ATTACK")
    status_color = "red" if attacks_detected > 0 else "green"
    status_text = "THREATS ACTIVE" if attacks_detected > 0 else "NETWORK NORMAL"
    st.markdown(
        f"""
        <div style="background: rgba(30,34,45,0.8); border: 1px solid #374151; padding: 12px; border-radius: 8px; text-align: center;">
            <small style="color: gray;">FLEET STATUS</small><br/>
            <strong style="color: {status_color}; font-size: 1.15rem;">● {status_text}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Top KPI Dashboard Cards
# ---------------------------------------------------------------------------
res = st.session_state.results
tot = len(res)
atk = sum(1 for r in res if r["Verdict"] == "ATTACK")
bng = tot - atk
det_rate = (atk / tot * 100) if tot > 0 else 0
latencies = [r["Latency (ms)"] for r in res]
avg_lat = np.mean(latencies) if latencies else 0.0
fps = int(1000 / avg_lat) if avg_lat > 0 else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Flows", f"{tot:,}")
kpi2.metric("Benign Flows", f"{bng:,}")
kpi3.metric("Attacks Intercepted", f"{atk:,}", delta=f"{atk} alerts" if atk > 0 else None, delta_color="inverse")
kpi4.metric("Threat Detection Rate", f"{det_rate:.1f}%")
kpi5.metric("Throughput", f"{fps:,} flows/s", delta=f"{avg_lat:.1f}ms lat")

# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------
tab_live, tab_bench, tab_pcap, tab_analytics = st.tabs([
    "⚡ Live Attack Simulation",
    "🧪 Dataset Benchmark Replay",
    "📁 PCAP File Analyzer",
    "📊 Model Analytics & Explainability",
])

# ===========================================================================
# TAB 1: Live Attack Simulation
# ===========================================================================
with tab_live:
    st.markdown("### 🎯 Synthetic Traffic Generator")
    st.caption("Craft live Scapy packets in memory, extract 52 flow statistics, and execute sub-millisecond inference.")

    # Preset scenario buttons
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    run_http = c1.button("🌐 Benign HTTP", use_container_width=True, help="Simulate normal HTTP REST temperature sensors")
    run_mqtt = c2.button("📡 Benign MQTT", use_container_width=True, help="Simulate normal MQTT telemetry publishing on port 1883")
    run_syn  = c3.button("⚠️ SYN Flood", use_container_width=True, help="Simulate aggressive 300-packet TCP SYN flood")
    run_udp  = c4.button("🌊 UDP Flood", use_container_width=True, help="Simulate high-bandwidth UDP packet blast")
    run_scan = c5.button("🔍 Port Scan", use_container_width=True, help="Simulate horizontal TCP SYN reconnaissance probe")
    run_mix  = c6.button("🔀 Mixed Fleet", use_container_width=True, help="Concurrent benign sensors + live attacker")

    base_time = time.time()

    if run_http:
        with st.spinner("Generating benign HTTP telemetry..."):
            pkts = gen_benign_http(base_time, n_flows=6)
            out = process_packets(loaded, pkts, "Benign HTTP")
            st.session_state.results.extend(out)
        st.success(f"Generated {len(pkts)} packets -> {len(out)} flows extracted and evaluated.")
        st.rerun()

    if run_mqtt:
        with st.spinner("Generating benign MQTT sensor traffic..."):
            pkts = gen_benign_mqtt(base_time, n_messages=6)
            out = process_packets(loaded, pkts, "Benign MQTT")
            st.session_state.results.extend(out)
        st.success(f"Generated {len(pkts)} packets -> {len(out)} flows extracted and evaluated.")
        st.rerun()

    if run_syn:
        with st.spinner("Injecting DDoS SYN flood attack..."):
            pkts = gen_syn_flood(base_time, n_packets=300)
            out = process_packets(loaded, pkts, "DDoS SYN Flood")
            st.session_state.results.extend(out)
        st.warning(f"Simulated {len(pkts)} attack packets -> {len(out)} flow analyzed.")
        st.rerun()

    if run_udp:
        with st.spinner("Injecting DDoS UDP bandwidth flood..."):
            pkts = gen_udp_flood(base_time, n_packets=300)
            out = process_packets(loaded, pkts, "DDoS UDP Flood")
            st.session_state.results.extend(out)
        st.warning(f"Simulated {len(pkts)} attack packets -> {len(out)} flow analyzed.")
        st.rerun()

    if run_scan:
        with st.spinner("Injecting reconnaissance port scan..."):
            pkts = gen_port_scan(base_time, n_ports=50)
            out = process_packets(loaded, pkts, "Recon Port Scan")
            st.session_state.results.extend(out)
        st.info(f"Simulated {len(pkts)} probe packets -> {len(out)} flows analyzed.")
        st.rerun()

    if run_mix:
        with st.spinner("Simulating multi-host enterprise IoT fleet..."):
            pkts = gen_mixed_fleet(base_time)
            out = process_packets(loaded, pkts, "Mixed Multi-Host Fleet")
            st.session_state.results.extend(out)
        st.info(f"Simulated {len(pkts)} packets across multiple hosts -> {len(out)} flows analyzed.")
        st.rerun()

    # Custom packet injection form
    with st.expander("🛠️ Custom Attack Injection Studio"):
        ci_col1, ci_col2, ci_col3, ci_col4 = st.columns(4)
        c_attack = ci_col1.selectbox("Attack Type", ["TCP SYN Flood", "UDP Flood", "Port Scan Sweep", "Mirai Telnet Brute Force"])
        c_src = ci_col2.text_input("Source IP (Attacker)", value="10.0.0.99")
        c_dst = ci_col3.text_input("Destination IP (IoT Hub)", value="10.0.0.20")
        c_pkts = ci_col4.slider("Packet Volume", min_value=50, max_value=800, value=250, step=50)

        if st.button("🚀 Launch Custom Simulation"):
            with st.spinner(f"Injecting {c_attack} ({c_pkts} packets)..."):
                if c_attack == "TCP SYN Flood":
                    pkts = gen_syn_flood(time.time(), n_packets=c_pkts, src_ip=c_src, dst_ip=c_dst)
                elif c_attack == "UDP Flood":
                    pkts = gen_udp_flood(time.time(), n_packets=c_pkts, src_ip=c_src, dst_ip=c_dst)
                elif c_attack == "Port Scan Sweep":
                    pkts = gen_port_scan(time.time(), n_ports=min(c_pkts, 100), src_ip=c_src, dst_ip=c_dst)
                else:
                    pkts = gen_mirai_botnet(time.time(), n_packets=c_pkts, src_ip=c_src, dst_ip=c_dst)

                out = process_packets(loaded, pkts, f"Custom {c_attack}")
                st.session_state.results.extend(out)
            st.success(f"Custom scenario executed: {len(out)} flows processed.")
            st.rerun()

    st.markdown("---")

    # Display Alerts and Flow Table
    col_alerts, col_table = st.columns([1, 2])

    with col_alerts:
        st.markdown("#### 🚨 Real-time Security Alerts")
        alerts = [r for r in st.session_state.results if r["Verdict"] == "ATTACK"]
        if alerts:
            for al in alerts[-8:]:
                sev_icon = "🔴" if al["Severity"] == "HIGH" else "🟠" if al["Severity"] == "MEDIUM" else "🟡"
                css_class = "alert-high" if al["Severity"] == "HIGH" else "alert-medium"
                st.markdown(f"""
                <div class="{css_class}">
                    <b>{sev_icon} [{al['Severity']}] {al['Class']}</b><br/>
                    <small><b>Flow:</b> <code>{al['Flow Key']}</code></small><br/>
                    <small>Time: {al['Time']} | Conf: <b>{al['Confidence']:.2%}</b> | Latency: {al['Latency (ms)']}ms</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No attack alerts triggered. Run an attack scenario to test detection.")

    with col_table:
        st.markdown("#### 📋 Flow Detection Log")
        if st.session_state.results:
            df_display = pd.DataFrame(st.session_state.results)
            cols = ["Time", "Scenario", "Flow Key", "Packets", "Verdict", "Class", "Confidence", "Severity", "Latency (ms)"]
            st.dataframe(df_display[cols], use_container_width=True, height=350)
        else:
            st.info("Click one of the scenario buttons above to simulate IoT traffic.")

    # Deep Flow Inspector
    if st.session_state.results:
        with st.expander("🔍 Deep Feature Inspector (Examine 52 Flow Features)"):
            flow_options = [f"Flow {i}: {r['Flow Key']} ({r['Verdict']} - {r['Class']})" for i, r in enumerate(st.session_state.results)]
            selected_flow_idx = st.selectbox("Select Flow to Inspect", range(len(flow_options)), format_func=lambda i: flow_options[i])
            chosen_flow = st.session_state.results[selected_flow_idx]

            raw_feats = chosen_flow.get("raw_features", {})
            feat_df = pd.DataFrame([{"Feature": k, "Value": v} for k, v in raw_feats.items()])
            st.dataframe(feat_df, use_container_width=True, height=260)


# ===========================================================================
# TAB 2: Dataset Benchmark Replay
# ===========================================================================
with tab_bench:
    st.markdown("### 🧪 Real CICIoT2023 Dataset Evaluation")
    st.caption("Replay ground-truth flows directly from the official test split of the CICIoT2023 dataset.")

    bench_df = get_benchmark_samples()
    if bench_df is not None:
        c_cat, c_count, c_action = st.columns([2, 1, 1])

        # Available labels/families
        all_labels = sorted(bench_df["label"].unique().tolist())
        all_labels.insert(0, "ALL_LABELS")

        chosen_label = c_cat.selectbox("Select Attack Category / Label", all_labels, index=0)
        n_samples = c_count.slider("Samples to Replay", min_value=10, max_value=min(500, len(bench_df)), value=50, step=10)

        if c_action.button("▶️ Run Evaluation", use_container_width=True):
            with st.spinner("Extracting features and scoring samples..."):
                subset = bench_df if chosen_label == "ALL_LABELS" else bench_df[bench_df["label"] == chosen_label]
                if len(subset) > n_samples:
                    subset = subset.sample(n_samples, random_state=42)

                bench_out = []
                for _, row in subset.iterrows():
                    feat_dict = {c: float(row.get(c, 0.0)) for c in loaded.features}
                    pred = predict_one(loaded, feat_dict)
                    true_label = str(row.get("label", "Unknown"))
                    true_is_attack = 0 if "benign" in true_label.lower() else 1

                    bench_out.append({
                        "True Label": true_label,
                        "Expected": "ATTACK" if true_is_attack == 1 else "BENIGN",
                        "Predicted": "ATTACK" if pred.get("prediction") == 1 else "BENIGN",
                        "Model Class": pred.get("label", ""),
                        "Match": "✅ CORRECT" if (pred.get("prediction") == true_is_attack) else "❌ MISSED",
                        "Confidence": round(float(pred.get("confidence", 0.0) or 0.0), 4),
                        "Latency (ms)": round(float(pred.get("inference_ms", 0.0)), 2),
                    })
                st.session_state.benchmark_results = bench_out
            st.success(f"Evaluated {len(bench_out)} benchmark samples!")

        if st.session_state.benchmark_results:
            b_df = pd.DataFrame(st.session_state.benchmark_results)

            correct = sum(1 for r in st.session_state.benchmark_results if "CORRECT" in r["Match"])
            total_b = len(st.session_state.benchmark_results)
            acc_score = (correct / total_b * 100) if total_b > 0 else 0

            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Samples Replayed", f"{total_b}")
            bc2.metric("Correct Predictions", f"{correct}")
            bc3.metric("Live Accuracy", f"{acc_score:.1f}%")

            st.dataframe(b_df, use_container_width=True, height=350)
    else:
        st.warning("No benchmark dataset file found at `data/demo_benchmark_sample.parquet`.")


# ===========================================================================
# TAB 3: PCAP File Analyzer
# ===========================================================================
with tab_pcap:
    st.markdown("### 📁 Custom Network Capture (.pcap) Analyzer")
    st.caption("Upload raw packet captures captured with Wireshark, tcpdump, or Zeek to run end-to-end classification.")

    uploaded_file = st.file_uploader("Upload PCAP / PCAPNG File", type=["pcap", "pcapng"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as fp:
            fp.write(uploaded_file.getbuffer())
            pcap_path = Path(fp.name)

        if st.button("⚡ Analyze PCAP File", use_container_width=True):
            with st.spinner("Extracting bidirectional 5-tuple flows and computing feature vectors..."):
                flows_df = pcap_to_flow_rows(pcap_path)
                pcap_path.unlink(missing_ok=True)

                pcap_results = []
                for idx, row in flows_df.iterrows():
                    feat = {c: float(row.get(c, 0.0)) for c in loaded.features}
                    pred = predict_one(loaded, feat)
                    is_attack = (pred.get("prediction") == 1)
                    conf = float(pred.get("confidence", 0.0) or 0.0)

                    pcap_results.append({
                        "Flow Key": str(row.get("_flow_key", f"flow_{idx}")),
                        "Packets": int(row.get("_packet_count", 0)),
                        "Verdict": "ATTACK" if is_attack else "BENIGN",
                        "Classification": pred.get("label", ""),
                        "Confidence": round(conf, 4),
                        "Inference (ms)": round(float(pred.get("inference_ms", 0.0)), 2),
                    })

            st.success(f"Extracted {len(pcap_results)} network flows from upload.")
            if pcap_results:
                p_df = pd.DataFrame(pcap_results)
                st.dataframe(p_df, use_container_width=True)


# ===========================================================================
# TAB 4: Model Analytics & Explainability
# ===========================================================================
with tab_analytics:
    st.markdown("### 📊 AI Model Architecture & Feature Importance")
    st.caption("Inspect algorithm properties, top split drivers, and dataset benchmarking.")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("#### Model Specification")
        spec_data = {
            "Property": ["Algorithm", "Task Type", "Feature Dimension", "Classification Threshold", "Model File Size"],
            "Value": [
                meta.get("model", Path(active_model_dir).name),
                loaded.task.capitalize(),
                f"{len(loaded.features)} input columns",
                str(loaded.threshold),
                f"{os.path.getsize(Path(active_model_dir) / 'model.joblib') / 1024:.1f} KB" if (Path(active_model_dir) / "model.joblib").exists() else "N/A"
            ]
        }
        st.table(pd.DataFrame(spec_data))

    with col_a2:
        st.markdown("#### Feature Importance Profile")
        if hasattr(loaded.model, "feature_importances_"):
            importances = loaded.model.feature_importances_
            if len(importances) == len(loaded.features):
                top_feats = pd.Series(importances, index=loaded.features).sort_values(ascending=False).head(10)
                fig = px.bar(
                    x=top_feats.values[::-1],
                    y=top_feats.index[::-1],
                    orientation="h",
                    labels={"x": "Relative Gini Importance", "y": "Feature"},
                    title="Top 10 Decision Drivers",
                    color=top_feats.values[::-1],
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importances not available for this model type.")

    st.markdown("---")
    st.markdown("#### 🏗️ End-to-End IDS Pipeline Architecture")
    st.markdown("""
    ```
    ┌──────────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐      ┌──────────────────────┐
    │  Raw Network Packets │ ───> │ Flow Aggregation Engine │ ───> │ Statistical Extractor  │ ───> │ Trained AI Classifier│
    │  (Ethernet/IP/TCP)   │      │ (Bidirectional 5-tuple) │      │ (52 CICIoT2023 Features)│     │ (Sub-ms inference)   │
    └──────────────────────┘      └─────────────────────────┘      └────────────────────────┘      └──────────────────────┘
                                                                                                               │
                                                                                                               ▼
                                                                                                   ┌──────────────────────┐
                                                                                                   │ Alert Feed & SOC KPI │
                                                                                                   │ (HIGH / MED / BENIGN)│
                                                                                                   └──────────────────────┘
    ```
    """)
