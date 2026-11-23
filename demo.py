#!/usr/bin/env python3
"""
CICIoT2023 IDS — Quick Demo Script
===================================

Proves your trained model works end-to-end:
  synthetic traffic  ->  flow extraction  ->  ML inference  ->  alerts

NO Docker required. NO network capture required. Runs in 10 seconds.

Usage:
    python demo.py                          # auto-finds model in models/binary/
    python demo.py --model-dir models/binary/XGBoost
    python demo.py --model-dir models/binary/XGBoost --scenarios all

What it does:
  1. Loads your trained model + preprocessing + label encoder
  2. Generates synthetic benign traffic (HTTP flows)
  3. Generates synthetic attack traffic (SYN flood)
  4. Extracts CICIoT2023-compatible flow features from each
  5. Runs the model on each flow
  6. Prints detection results + alerts

This is a PROOF OF CONCEPT — the traffic is synthetic but the detection
chain is real: pcap -> flow features -> trained model -> alert.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add repo to path if running from repo root
REPO = Path(__file__).resolve().parent
if (REPO / "src").exists():
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

# Suppress scapy's noisy MAC-address warnings (we never actually send packets)
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
try:
    from scapy.all import conf
    conf.verb = 0
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Colors for terminal output
# ---------------------------------------------------------------------------
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
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
# Step 1: Load the trained model
# ---------------------------------------------------------------------------
def load_model(model_dir: str):
    """Load model + preprocessing + features from a model directory."""
    from src.inference.model_loader import load_model_dir

    p = Path(model_dir)
    if not (p / "model.joblib").exists():
        print(bad(f"ERROR: No model.joblib found in {p}"))
        print(f"  Looking for: {p / 'model.joblib'}")
        print(f"  Train a model first, or specify --model-dir")
        sys.exit(1)

    loaded = load_model_dir(p)
    model_name = loaded.metadata.get('model') or p.name
    print(ok(f"[loaded] model={model_name}  "
             f"task={loaded.task}  features={len(loaded.features)}  "
             f"threshold={loaded.threshold}"))
    if loaded.metadata.get("metrics"):
        m = loaded.metadata["metrics"]
        f1_m = m.get('f1_macro', m.get('f1', '?'))
        fpr_m = m.get('fpr', m.get('FPR', '?'))
        fnr_m = m.get('fnr', m.get('FNR', '?'))
        print(f"         macro_f1={f1_m}  "
              f"fpr={fpr_m}  fnr={fnr_m}")
    return loaded


def auto_find_model() -> str:
    """Find the first available trained model in models/binary/."""
    for subdir in ["binary", "multiclass"]:
        base = Path("models") / subdir
        if not base.exists():
            continue
        # Prefer RandomForest, then XGBoost, then LightGBM, then anything
        for name in ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression", "DecisionTree"]:
            p = base / name
            if (p / "model.joblib").exists():
                return str(p)
        # If no preferred name, take first available
        for p in sorted(base.iterdir()):
            if (p / "model.joblib").exists():
                return str(p)
    # Check if we're in a subdirectory
    for root in [Path("."), Path(".."), Path("/content/ciciot2023-ids")]:
        for subdir in ["binary", "multiclass"]:
            base = root / "models" / subdir
            if not base.exists():
                continue
            for name in ["RandomForest", "XGBoost", "LightGBM"]:
                p = base / name
                if (p / "model.joblib").exists():
                    return str(p)
    return ""


# ---------------------------------------------------------------------------
# Step 2: Generate synthetic traffic (benign + attack)
# ---------------------------------------------------------------------------
def make_packet(src, dst, sport, dport, flags, ts, payload=None):
    """Build a single scapy packet."""
    from scapy.all import IP, TCP, Ether, Raw
    if payload:
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags) / Raw(load=payload)
    else:
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    p.time = ts
    return p


def generate_benign_traffic(base_time: float, n_flows: int = 8) -> List:
    """Generate benign HTTP-like flows: SYN -> SYN-ACK -> ACK -> GET -> ACK."""
    pkts = []
    for f in range(n_flows):
        sport = 50000 + f
        t = base_time + f * 0.3
        # 3-way handshake + HTTP GET
        pkts.append(make_packet("10.0.0.10", "10.0.0.20", sport, 80, "S", t))
        pkts.append(make_packet("10.0.0.20", "10.0.0.10", 80, sport, "SA", t + 0.01))
        pkts.append(make_packet("10.0.0.10", "10.0.0.20", sport, 80, "A", t + 0.02))
        pkts.append(make_packet("10.0.0.10", "10.0.0.20", sport, 80, "PA", t + 0.03,
                                payload=f"GET /sensor/{f} HTTP/1.1\r\nHost: iot.local\r\n\r\n"))
        pkts.append(make_packet("10.0.0.20", "10.0.0.10", 80, sport, "A", t + 0.04))
    return pkts


def generate_attack_traffic(base_time: float, attack_type: str = "syn_flood",
                             n_packets: int = 300) -> List:
    """Generate attack-like traffic: high-rate SYN flood in a single flow."""
    pkts = []
    if attack_type == "syn_flood":
        # DDoS-SYN-Flood: one 5-tuple, many SYN packets, very high rate
        for i in range(n_packets):
            pkts.append(make_packet("10.0.0.99", "10.0.0.20", 40000, 80, "S",
                                    base_time + i * 0.0001))
    elif attack_type == "udp_flood":
        # DDoS-UDP-Flood: high-rate UDP to a single port
        from scapy.all import IP, UDP, Ether
        for i in range(n_packets):
            p = Ether() / IP(src="10.0.0.99", dst="10.0.0.20") / UDP(sport=40000, dport=1234)
            p.time = base_time + i * 0.0001
            pkts.append(p)
    elif attack_type == "port_scan":
        # Recon-Port_Scan: SYN to many different ports
        for i in range(n_packets):
            pkts.append(make_packet("10.0.0.99", "10.0.0.20", 40000, 1 + i, "S",
                                    base_time + i * 0.01))
    return pkts


# ---------------------------------------------------------------------------
# Step 3: Extract flow features and run inference
# ---------------------------------------------------------------------------
def run_detection(loaded, packets: List, scenario_name: str) -> List[Dict]:
    """Extract flows from packets, run the model on each, return results."""
    from scapy.all import wrpcap
    from src.simulation.flow_extraction import pcap_to_flow_rows
    from src.inference.predictor import predict_one

    # Write packets to a temp pcap, then extract flows
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".pcap"))
    wrpcap(str(tmp), packets)

    # Extract flow features (this is the real CICIoT2023-compatible extraction)
    df = pcap_to_flow_rows(tmp)
    tmp.unlink()

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
            "attack_probability": pred.get("attack_probability"),
        })
    return results


# ---------------------------------------------------------------------------
# Step 4: Pretty-print results
# ---------------------------------------------------------------------------
def print_alert(result: Dict):
    """Print an alert in the IDS alert format."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if result["prediction"] == 1:
        print(bad(f"""
┌─ [ALERT] ──────────────────────────────────────────────
│ Time       : {ts}
│ Detection  : ATTACK
│ Class      : {result['label']}
│ Confidence : {result['confidence']:.4f}
│ Flow       : {result['flow_key']}
│ Packets    : {result['packets']}
│ Severity   : {"HIGH" if result['confidence'] > 0.85 else "MEDIUM" if result['confidence'] > 0.65 else "LOW"}
│ Inference  : {result['inference_ms']:.3f} ms
└──────────────────────────────────────────────────────────"""))
    else:
        print(ok(f"  [OK]  {result['flow_key']:<45} pkts={result['packets']:>4}  "
                 f"conf={result['confidence']:.4f}  {result['inference_ms']:.3f}ms"))


def print_summary(all_results: List[Dict]):
    """Print a final summary table."""
    total = len(all_results)
    attacks = sum(1 for r in all_results if r["prediction"] == 1)
    benign = total - attacks
    avg_inf = np.mean([r["inference_ms"] for r in all_results]) if all_results else 0

    print(f"\n{'='*60}")
    print(header("  DEMO SUMMARY"))
    print(f"{'='*60}")
    print(f"  Total flows analyzed : {total}")
    print(f"  {ok('Benign detected'):<25} : {benign}")
    print(f"  {bad('Attacks detected'):<25} : {attacks}")
    print(f"  Detection rate        : {100*attacks/max(total,1):.1f}%")
    print(f"  Avg inference time    : {avg_inf:.3f} ms/flow")
    print(f"  Throughput            : {1000/max(avg_inf,0.001):,.0f} flows/sec")
    print(f"{'='*60}")

    if attacks > 0:
        print(f"\n{bad('⚠  ATTACKS DETECTED — IDS is working!')}")
    else:
        print(f"\n{ok('All traffic classified as benign.')}")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def run_demo(model_dir: str, scenarios: str = "all"):
    """Run the full demo."""
    print(f"\n{'='*60}")
    print(header("  CICIoT2023 AI-BASED IoT IDS — LIVE DEMO"))
    print(f"{'='*60}\n")

    # Step 1: Load model
    print(header("STEP 1: Loading trained model"))
    print(f"  {C.DIM}Model directory: {model_dir}{C.END}")
    loaded = load_model(model_dir)
    print()

    # Step 2: Generate + detect benign traffic
    all_results = []

    if scenarios in ("all", "benign"):
        print(header("STEP 2: Scenario A — Benign IoT Traffic"))
        print(f"  {C.DIM}Generating 8 normal HTTP flows (temperature sensor polling)...{C.END}")
        t0 = time.time()
        benign_pkts = generate_benign_traffic(base_time=time.time(), n_flows=8)
        print(f"  Generated {len(benign_pkts)} packets in {time.time()-t0:.2f}s")
        print(f"  {C.DIM}Extracting flow features and running inference...{C.END}\n")
        results = run_detection(loaded, benign_pkts, "benign")
        for r in results:
            print_alert(r)
        all_results.extend(results)
        print()

    # Step 3: Generate + detect attack traffic
    if scenarios in ("all", "attack"):
        print(header("STEP 3: Scenario B — DDoS SYN Flood Attack"))
        print(f"  {C.DIM}Generating SYN flood: 300 packets in 1 flow at ~10,000 pps...{C.END}")
        t0 = time.time()
        attack_pkts = generate_attack_traffic(
            base_time=time.time() + 10, attack_type="syn_flood", n_packets=300)
        print(f"  Generated {len(attack_pkts)} attack packets in {time.time()-t0:.2f}s")
        print(f"  {C.DIM}Extracting flow features and running inference...{C.END}\n")
        results = run_detection(loaded, attack_pkts, "syn_flood")
        for r in results:
            print_alert(r)
        all_results.extend(results)
        print()

    # Step 4: Mixed scenario
    if scenarios in ("all", "mixed"):
        print(header("STEP 4: Scenario C — Mixed Traffic (benign + attack simultaneously)"))
        base = time.time() + 20
        benign = generate_benign_traffic(base_time=base, n_flows=5)
        attack = generate_attack_traffic(base_time=base + 0.5, attack_type="syn_flood", n_packets=200)
        mixed = benign + attack
        print(f"  {C.DIM}Generated {len(benign)} benign + {len(attack)} attack = {len(mixed)} total packets{C.END}")
        print(f"  {C.DIM}Extracting flow features and running inference...{C.END}\n")
        results = run_detection(loaded, mixed, "mixed")
        for r in results:
            print_alert(r)
        all_results.extend(results)
        print()

    # Step 5: Summary
    print_summary(all_results)

    # Save results
    out = Path("demo_results.json")
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n{C.DIM}Results saved to {out}{C.END}")

    return all_results


def main():
    ap = argparse.ArgumentParser(description="CICIoT2023 IDS Quick Demo")
    ap.add_argument("--model-dir", default=None,
                    help="path to model directory (auto-detected if omitted)")
    ap.add_argument("--scenarios", default="all",
                    choices=["all", "benign", "attack", "mixed"],
                    help="which scenarios to run")
    args = ap.parse_args()

    model_dir = args.model_dir or auto_find_model()
    if not model_dir:
        print(bad("ERROR: No trained model found."))
        print("  Either:")
        print("    1. Run from the repo root (models/binary/XGBoost/ should exist)")
        print("    2. Specify --model-dir /path/to/model")
        sys.exit(1)

    run_demo(model_dir, args.scenarios)


if __name__ == "__main__":
    main()
