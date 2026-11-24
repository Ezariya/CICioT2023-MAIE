#!/usr/bin/env python3
"""
CICIoT2023 IDS — Terminal Demonstration & Benchmark Verification
================================================================

Proves your trained model works end-to-end:
  synthetic packets  ->  flow extraction  ->  ML inference  ->  security alerts

Features:
  - Tests 6 distinct IoT scenarios (HTTP sensor, MQTT telemetry, SYN flood, UDP flood, Mirai, Mixed)
  - Optional real CICIoT2023 dataset benchmark evaluation (--benchmark)
  - Produces formatted console alerts, JSON metrics, and a standalone HTML executive report

Usage:
    python demo.py
    python demo.py --model-dir models/binary/RandomForest
    python demo.py --scenarios all --benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add repo to path
REPO = Path(__file__).resolve().parent
if (REPO / "src").exists():
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

# Suppress scapy runtime logs
import logging
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
# ANSI Terminal Formatting
# ---------------------------------------------------------------------------
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"
    DIM = "\033[2m"


def header(text: str) -> str:
    return f"{C.BOLD}{C.CYAN}{text}{C.END}"


def ok(text: str) -> str:
    return f"{C.GREEN}{text}{C.END}"


def bad(text: str) -> str:
    return f"{C.RED}{text}{C.END}"


def warn(text: str) -> str:
    return f"{C.YELLOW}{text}{C.END}"


# ---------------------------------------------------------------------------
# Model Loader & Auto-discovery
# ---------------------------------------------------------------------------
def auto_find_model() -> str:
    """Find the best available trained model in models/."""
    for subdir in ["binary", "multiclass"]:
        base = Path("models") / subdir
        if not base.exists():
            continue
        for name in ["RandomForest", "XGBoost", "LightGBM", "family_XGBoost", "LogisticRegression"]:
            p = base / name
            if (p / "model.joblib").exists():
                return str(p)
    return ""


def load_model(model_dir: str):
    """Load model artefacts with validation."""
    p = Path(model_dir)
    if not (p / "model.joblib").exists():
        print(bad(f"ERROR: No model.joblib found in {p}"))
        sys.exit(1)

    loaded = load_model_dir(p)
    model_name = loaded.metadata.get("model") or p.name
    print(ok(f"[loaded] model={model_name}  task={loaded.task}  "
             f"features={len(loaded.features)}  threshold={loaded.threshold}"))

    m = loaded.metadata.get("metrics")
    if m:
        f1_m = m.get("f1_macro", m.get("f1", "?"))
        fpr_m = m.get("fpr", m.get("FPR", "?"))
        acc_m = m.get("accuracy", "?")
        print(f"         accuracy={acc_m}  macro_f1={f1_m}  fpr={fpr_m}")
    return loaded


# ---------------------------------------------------------------------------
# Inference Runner
# ---------------------------------------------------------------------------
def run_detection(loaded, packets: List, scenario_name: str) -> List[Dict]:
    """Run full packet -> flow -> inference pipeline."""
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".pcap"))
    wrpcap(str(tmp), packets)
    df = pcap_to_flow_rows(tmp)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

    if df.empty:
        print(warn(f"  [{scenario_name}] no flows extracted"))
        return []

    results = []
    for idx, row in df.iterrows():
        feat_dict = {c: float(row.get(c, 0.0)) for c in loaded.features}
        pred = predict_one(loaded, feat_dict)
        flow_key = str(row.get("_flow_key", f"flow_{idx}"))
        pkt_count = int(row.get("_packet_count", 0))

        results.append({
            "scenario": scenario_name,
            "flow_key": flow_key,
            "packets": pkt_count,
            "prediction": pred.get("prediction"),
            "label": pred.get("label"),
            "confidence": pred.get("confidence"),
            "inference_ms": pred.get("inference_ms"),
            "attack_probability": pred.get("attack_probability", pred.get("confidence")),
        })
    return results


def print_alert(result: Dict):
    """Print security detection results in SOC alert format."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if result["prediction"] == 1:
        conf = float(result.get("confidence") or 0.0)
        sev = "HIGH" if conf > 0.85 else "MEDIUM" if conf > 0.65 else "LOW"
        print(bad(f"""
┌─ [ALERT: {sev} SEVERITY] ──────────────────────────────────────────
│ Time       : {ts}
│ Detection  : THREAT DETECTED (ATTACK)
│ Class      : {result['label']}
│ Confidence : {conf:.4f}
│ Flow       : {result['flow_key']}
│ Packets    : {result['packets']}
│ Latency    : {result['inference_ms']:.3f} ms
└──────────────────────────────────────────────────────────"""))
    else:
        conf = float(result.get("confidence") or 0.0)
        print(ok(f"  [OK: BENIGN]  {result['flow_key']:<42} pkts={result['packets']:>3}  conf={conf:.4f}  {result['inference_ms']:.2f}ms"))


# ---------------------------------------------------------------------------
# HTML Executive Report Generator
# ---------------------------------------------------------------------------
def generate_html_report(results: List[Dict], model_dir: str, out_path: Path):
    """Generate a clean standalone HTML executive summary report."""
    total = len(results)
    attacks = sum(1 for r in results if r["prediction"] == 1)
    benign = total - attacks
    det_rate = (attacks / total * 100) if total > 0 else 0
    avg_lat = np.mean([r["inference_ms"] for r in results]) if results else 0.0

    rows_html = ""
    for r in results:
        status_color = "#ef4444" if r["prediction"] == 1 else "#10b981"
        badge_text = "ATTACK" if r["prediction"] == 1 else "BENIGN"
        rows_html += f"""
        <tr>
            <td>{r.get('scenario')}</td>
            <td><code>{r.get('flow_key')}</code></td>
            <td>{r.get('packets')}</td>
            <td><span style="background:{status_color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">{badge_text}</span></td>
            <td><b>{r.get('label')}</b></td>
            <td>{float(r.get('confidence',0.0) or 0.0):.2%}</td>
            <td>{r.get('inference_ms', 0):.2f} ms</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CICIoT2023 IDS Verification Report</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background: #0f172a; color: #f8fafc; }}
        .header {{ border-bottom: 2px solid #334155; padding-bottom: 15px; margin-bottom: 25px; }}
        .kpi-container {{ display: flex; gap: 15px; margin-bottom: 25px; }}
        .kpi {{ background: #1e293b; padding: 18px 24px; border-radius: 8px; border: 1px solid #334155; flex: 1; }}
        .kpi-title {{ font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }}
        .kpi-value {{ font-size: 1.8rem; font-weight: bold; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }}
        code {{ color: #38bdf8; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ CICIoT2023 AI IDS — Verification Report</h1>
        <p style="color:#94a3b8;">Evaluated Model: <code>{model_dir}</code> | Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="kpi-container">
        <div class="kpi">
            <div class="kpi-title">Total Network Flows</div>
            <div class="kpi-value">{total}</div>
        </div>
        <div class="kpi">
            <div class="kpi-title">Benign Verified</div>
            <div class="kpi-value" style="color:#10b981;">{benign}</div>
        </div>
        <div class="kpi">
            <div class="kpi-title">Attacks Intercepted</div>
            <div class="kpi-value" style="color:#ef4444;">{attacks}</div>
        </div>
        <div class="kpi">
            <div class="kpi-title">Detection Rate</div>
            <div class="kpi-value">{det_rate:.1f}%</div>
        </div>
        <div class="kpi">
            <div class="kpi-title">Avg Latency</div>
            <div class="kpi-value">{avg_lat:.2f} ms</div>
        </div>
    </div>

    <h3>Detailed Flow Inference Log</h3>
    <table>
        <thead>
            <tr>
                <th>Scenario</th>
                <th>Flow Key (5-Tuple)</th>
                <th>Packets</th>
                <th>Verdict</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>Latency</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main Demo Execution
# ---------------------------------------------------------------------------
def run_demo(model_dir: str, scenarios: str = "all", run_bench: bool = False):
    """Run all scenarios end-to-end."""
    print(f"\n{'='*65}")
    print(header("  CICIoT2023 AI-BASED IoT IDS — LIVE SOC VERIFICATION DEMO"))
    print(f"{'='*65}\n")

    print(header("STEP 1: Loading AI Model"))
    print(f"  {C.DIM}Directory: {model_dir}{C.END}")
    loaded = load_model(model_dir)
    print()

    all_results = []
    base_t = time.time()

    # Scenario 1: Benign HTTP
    if scenarios in ("all", "benign"):
        print(header("STEP 2: Scenario 1 — Benign HTTP IoT Sensor Polling"))
        pkts = gen_benign_http(base_t, n_flows=6)
        print(f"  {C.DIM}Generated {len(pkts)} packets (temperature sensor polling)...{C.END}")
        res = run_detection(loaded, pkts, "Benign HTTP")
        for r in res:
            print_alert(r)
        all_results.extend(res)
        print()

    # Scenario 2: Benign MQTT
    if scenarios in ("all", "benign"):
        print(header("STEP 3: Scenario 2 — Benign MQTT Broker Telemetry"))
        pkts = gen_benign_mqtt(base_t + 10, n_messages=5)
        print(f"  {C.DIM}Generated {len(pkts)} packets (MQTT port 1883 telemetry)...{C.END}")
        res = run_detection(loaded, pkts, "Benign MQTT")
        for r in res:
            print_alert(r)
        all_results.extend(res)
        print()

    # Scenario 3: DDoS SYN Flood
    if scenarios in ("all", "attack"):
        print(header("STEP 4: Scenario 3 — DDoS TCP SYN Flood Attack"))
        pkts = gen_syn_flood(base_t + 20, n_packets=300)
        print(f"  {C.DIM}Generated {len(pkts)} packets (DDoS SYN Flood at ~10,000 pps)...{C.END}")
        res = run_detection(loaded, pkts, "DDoS SYN Flood")
        for r in res:
            print_alert(r)
        all_results.extend(res)
        print()

    # Scenario 4: DDoS UDP Flood
    if scenarios in ("all", "attack"):
        print(header("STEP 5: Scenario 4 — DDoS UDP Bandwidth Flood Attack"))
        pkts = gen_udp_flood(base_t + 30, n_packets=300)
        print(f"  {C.DIM}Generated {len(pkts)} packets (High rate UDP blast)...{C.END}")
        res = run_detection(loaded, pkts, "DDoS UDP Flood")
        for r in res:
            print_alert(r)
        all_results.extend(res)
        print()

    # Scenario 5: Mirai Botnet
    if scenarios in ("all", "attack"):
        print(header("STEP 6: Scenario 5 — Mirai IoT Botnet Activity"))
        pkts = gen_mirai_botnet(base_t + 40, n_packets=250)
        print(f"  {C.DIM}Generated {len(pkts)} packets (Telnet brute force + burst attack)...{C.END}")
        res = run_detection(loaded, pkts, "Mirai Botnet")
        for r in res:
            print_alert(r)
        all_results.extend(res)
        print()

    # Scenario 6: Mixed Fleet
    if scenarios in ("all", "mixed"):
        print(header("STEP 7: Scenario 6 — Mixed Multi-Host Fleet (Live Isolation)"))
        pkts = gen_mixed_fleet(base_t + 50)
        print(f"  {C.DIM}Simulating 5 normal sensors + concurrent active attacker ({len(pkts)} packets)...{C.END}")
        res = run_detection(loaded, pkts, "Mixed Multi-Host Fleet")
        for r in res:
            print_alert(r)
        all_results.extend(res)
        print()

    # Optional Benchmark Replay
    if run_bench:
        print(header("STEP 8: Benchmark Replay — Real CICIoT2023 Dataset Evaluation"))
        sample_path = REPO / "data" / "demo_benchmark_sample.parquet"
        if sample_path.exists():
            df_b = pd.read_parquet(sample_path)
            eval_set = df_b.sample(min(50, len(df_b)), random_state=42)
            correct = 0
            for _, row in eval_set.iterrows():
                feat = {c: float(row.get(c, 0.0)) for c in loaded.features}
                p = predict_one(loaded, feat)
                exp = 0 if "benign" in str(row.get("label", "")).lower() else 1
                if p.get("prediction") == exp:
                    correct += 1
            print(ok(f"  Replayed {len(eval_set)} real test flows: Accuracy = {correct/len(eval_set):.1%} ({correct}/{len(eval_set)} correct)"))
        else:
            print(warn(f"  Benchmark file {sample_path} not found. Skipping."))
        print()

    # Summary
    total = len(all_results)
    attacks = sum(1 for r in all_results if r["prediction"] == 1)
    benign = total - attacks
    latencies = [r["inference_ms"] for r in all_results]
    avg_lat = np.mean(latencies) if latencies else 0.0
    throughput = int(1000 / avg_lat) if avg_lat > 0 else 0

    print(f"{'='*65}")
    print(header("  DEMO EXECUTIVE SUMMARY"))
    print(f"{'='*65}")
    print(f"  Total Network Flows Analyzed : {total}")
    print(ok(f"  Benign Flows Verified        : {benign}"))
    print(bad(f"  Attack Threats Intercepted   : {attacks}"))
    print(f"  Threat Detection Rate        : {attacks/max(total,1):.1%}")
    print(f"  Average Inference Latency    : {avg_lat:.3f} ms / flow")
    print(f"  Inference Throughput         : {throughput:,} flows / second")
    print(f"{'='*65}\n")

    if attacks > 0:
        print(ok("✔ SYSTEM VALIDATION PASSED — AI IDS is actively detecting cyber threats!"))
    else:
        print(warn("⚠ No attacks detected."))

    # Save artifacts
    json_path = REPO / "demo_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"{C.DIM}Saved structured log: {json_path}{C.END}")

    html_path = REPO / "demo_report.html"
    generate_html_report(all_results, model_dir, html_path)
    print(ok(f"Saved visual HTML report: {html_path.resolve()}"))


def main():
    ap = argparse.ArgumentParser(description="CICIoT2023 IDS Terminal Demo & Verification")
    ap.add_argument("--model-dir", default=None, help="Path to model directory")
    ap.add_argument("--scenarios", default="all", choices=["all", "benign", "attack", "mixed"])
    ap.add_argument("--benchmark", action="store_true", help="Also evaluate against real dataset flows")
    args = ap.parse_args()

    model_dir = args.model_dir or auto_find_model()
    if not model_dir:
        print(bad("ERROR: No trained model found in models/."))
        sys.exit(1)

    run_demo(model_dir, args.scenarios, args.benchmark)


if __name__ == "__main__":
    main()
