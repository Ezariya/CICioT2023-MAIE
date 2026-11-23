"""
src/simulation/simulator.py — end-to-end live IDS simulator.

  Network
      |
      v
  Traffic (captured from isolated Docker net OR generated locally)
      |
      v
  Flow features (flow_extraction.pcap_to_flow_rows)
      |
      v
  Trained ML model (inference.predictor)
      |
      v
  Alert (simulation.alert)

Two modes:
  --live          : sniff from a real interface and run detection continuously
  --pcap PATH     : read an existing pcap, build flows, run the model on each
  --generate      : generate synthetic benign + attack-like traffic INSIDE an
                    isolated Docker network and capture it

Hard rule (spec section 25): NO fake "row = random.choice(test_dataset)".
The simulator always converts packets -> flows -> features -> model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils import load_config, project_root
from src.inference.model_loader import load_model_dir
from src.inference.predictor import predict_one
from src.simulation.traffic_capture import capture
from src.simulation.flow_extraction import pcap_to_flow_rows, write_mapping_report
from src.simulation.alert import AlertLogger


def run_on_pcap(pcap_path: Path, model_dir: Path,
                alert_log: Optional[Path] = None) -> List[Dict]:
    """Read pcap -> flows -> predict per flow -> log alerts."""
    loaded = load_model_dir(model_dir)
    df = pcap_to_flow_rows(pcap_path)
    if df.empty:
        print(f"[simulator] no flows extracted from {pcap_path}")
        return []
    logger = AlertLogger(log_path=alert_log)
    results: List[Dict] = []
    for _, row in df.iterrows():
        feat_dict = {c: row.get(c, 0.0) for c in loaded.features}
        pred = predict_one(loaded, feat_dict)
        alert = logger.log(pred, row.to_dict())
        results.append({
            "flow_key": row.get("_flow_key", "?"),
            "packet_count": int(row.get("_packet_count", 0)),
            "prediction": pred.get("prediction"),
            "label": pred.get("label"),
            "confidence": pred.get("confidence"),
            "inference_ms": pred.get("inference_ms"),
            "alert": alert.format() if alert else None,
        })
    logger.close()
    return results


def run_live(model_dir: Path, duration_sec: int, interface: str | None = None) -> List[Dict]:
    """Capture for `duration_sec` then run the model on the captured pcap."""
    pcap = capture(timeout_sec=duration_sec, interface=interface)
    return run_on_pcap(pcap, model_dir)


def generate_synthetic_traffic(out_pcap: Path,
                               benign_pkts: int = 200,
                               attack_pkts: int = 500) -> Path:
    """
    Generate a small synthetic pcap that contains:
      - benign TCP handshakes + HTTP GETs (low rate, multi-packet flows)
      - "attack-like" high-rate SYN flood from a spoofed source (multi-packet
        flow with very high syn_count and Rate)
    ONLY for use inside an isolated Docker network.
    """
    from scapy.all import IP, TCP, UDP, Ether, wrpcap, Raw
    out_pcap.parent.mkdir(parents=True, exist_ok=True)
    pkts = []
    base = time.time()

    # benign traffic: a few multi-packet TCP flows (SYN, SYN-ACK, ACK, GET)
    n_flows = max(1, benign_pkts // 5)
    for f in range(n_flows):
        sport = 50000 + f
        flow_base = base + f * 0.5
        # SYN
        pkts.append(_make_pkt("10.0.0.10", "10.0.0.20", sport, 80, "S",
                              flow_base, payload=None))
        # SYN-ACK
        pkts.append(_make_pkt("10.0.0.20", "10.0.0.10", 80, sport, "SA",
                              flow_base + 0.01, payload=None))
        # ACK
        pkts.append(_make_pkt("10.0.0.10", "10.0.0.20", sport, 80, "A",
                              flow_base + 0.02, payload=None))
        # HTTP GET
        pkts.append(_make_pkt("10.0.0.10", "10.0.0.20", sport, 80, "PA",
                              flow_base + 0.03,
                              payload=f"GET / HTTP/1.1\r\nHost: example.local\r\n\r\n"))
        # ACK
        pkts.append(_make_pkt("10.0.0.20", "10.0.0.10", 80, sport, "A",
                              flow_base + 0.04, payload=None))

    # attack-like traffic: ONE flow with a very high packet rate.
    # We use the SAME source port for every packet so they collapse into a
    # single 5-tuple flow with a high packet count + high syn_count + high Rate.
    attack_base = base + n_flows * 0.5 + 1
    for i in range(attack_pkts):
        pkts.append(_make_pkt("10.0.0.99", "10.0.0.20", 40000, 80, "S",
                              attack_base + i * 0.0001, payload=None))

    wrpcap(str(out_pcap), pkts)
    print(f"[simulator] wrote synthetic pcap with {len(pkts)} packets -> {out_pcap}")
    return out_pcap


def _make_pkt(src: str, dst: str, sport: int, dport: int, flags: str,
              ts: float, payload=None):
    """Build a TCP/IP packet with a given timestamp."""
    from scapy.all import IP, TCP, Ether, Raw
    if payload is not None:
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags) / Raw(load=payload)
    else:
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    p.time = ts
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pcap", default=None)
    ap.add_argument("--generate", action="store_true",
                    help="generate a synthetic pcap (isolated lab only)")
    ap.add_argument("--live", action="store_true",
                    help="live capture from the configured interface")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--interface", default=None)
    ap.add_argument("--out", default=None,
                    help="where to write the detection results JSON")
    args = ap.parse_args()

    if args.generate:
        out = Path(args.out) if args.out else (
            project_root() / "simulation" / "traffic" / f"synthetic_{int(time.time())}.pcap"
        )
        pcap_path = generate_synthetic_traffic(out)
        results = run_on_pcap(pcap_path, Path(args.model_dir))
    elif args.live:
        results = run_live(Path(args.model_dir), args.duration, args.interface)
    elif args.pcap:
        results = run_on_pcap(Path(args.pcap), Path(args.model_dir))
    else:
        ap.error("must give --pcap / --live / --generate")

    out_json = project_root() / "reports" / f"simulator_results_{int(time.time())}.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    n_attack = sum(1 for r in results if r["prediction"] == 1)
    print(f"[simulator] {len(results)} flows, {n_attack} attacks -> {out_json}")


if __name__ == "__main__":
    main()
