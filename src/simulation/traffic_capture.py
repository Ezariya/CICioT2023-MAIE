"""
src/simulation/traffic_capture.py — packet capture from a live interface.

Hard rule (spec section 25): NEVER replay a CSV row and call it live
detection. The simulation must capture real packets, build real flows, run
the trained model on the resulting flow features.

Two backends:
  - scapy:  pure-python sniff + craft (default; works everywhere)
  - pyshark: tshark wrapper (more features, needs tshark installed)

Only run inside the isolated Docker network defined in simulation/docker.
NO attack traffic to external networks.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

from src.utils import load_config, project_root


def capture_with_scapy(interface: str, timeout_sec: int, out_pcap: Path) -> Path:
    """Live-capture packets using scapy and write a pcap."""
    from scapy.all import sniff
    print(f"[capture] scapy sniff iface={interface} timeout={timeout_sec}s")
    pkts = sniff(iface=interface, timeout=timeout_sec)
    from scapy.all import wrpcap
    wrpcap(str(out_pcap), pkts)
    print(f"[capture] wrote {len(pkts)} packets to {out_pcap}")
    return out_pcap


def capture_with_pyshark(interface: str, timeout_sec: int, out_pcap: Path) -> Path:
    """Live-capture using pyshark (tshark wrapper)."""
    try:
        import pyshark
    except ImportError as e:
        raise RuntimeError("pyshark not installed; install with: pip install pyshark") from e
    print(f"[capture] pyshark live iface={interface} timeout={timeout_sec}s")
    cap = pyshark.LiveCapture(interface=interface, output_file=str(out_pcap))
    cap.sniff(timeout=timeout_sec)
    print(f"[capture] wrote pcap to {out_pcap}")
    return out_pcap


def capture(timeout_sec: int | None = None, interface: str | None = None,
            backend: str | None = None, out_pcap: Path | None = None) -> Path:
    cfg = load_config()
    sim_cfg = cfg["simulation"]
    timeout_sec = timeout_sec or sim_cfg["capture"]["timeout_sec"]
    interface = interface or sim_cfg["capture"]["interface"]
    backend = backend or sim_cfg["capture"]["backend"]
    if out_pcap is None:
        out_pcap = project_root() / "simulation" / "traffic" / f"capture_{int(time.time())}.pcap"
    out_pcap.parent.mkdir(parents=True, exist_ok=True)
    if backend == "scapy":
        return capture_with_scapy(interface, timeout_sec, out_pcap)
    if backend == "pyshark":
        return capture_with_pyshark(interface, timeout_sec, out_pcap)
    raise ValueError(f"unknown capture backend: {backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=None,
                    help="capture duration in seconds (default: config)")
    ap.add_argument("--interface", default=None)
    ap.add_argument("--backend", default=None, choices=["scapy", "pyshark"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    p = capture(timeout_sec=args.duration, interface=args.interface,
                backend=args.backend,
                out_pcap=Path(args.out) if args.out else None)
    print(f"[capture] done -> {p}")


if __name__ == "__main__":
    main()
