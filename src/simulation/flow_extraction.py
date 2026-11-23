"""
src/simulation/flow_extraction.py — packet -> flow -> CICIoT2023-compatible features.

This is the bridge between raw packets and the ML model (spec section 24).

  Packets
      |
      v
  Flow / session aggregation  (5-tuple: src_ip, dst_ip, src_port, dst_port, proto)
      |
      v
  Feature calculation         (statistical features over the flow window)
      |
      v
  Same feature schema as training (config/features.yaml)
      |
      v
  ML model

IMPORTANT (spec section 25):
If exact CICIoT2023 feature reproduction proves difficult, we document the
limitation and create a CONTROLLED feature-compatible layer rather than
pretending it is identical. The mapping table is written to
reports/flow_feature_mapping.json.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import load_config, project_root, write_json
from src.features.schema import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Flow aggregation
# ---------------------------------------------------------------------------
FLOW_KEYS = ("src_ip", "dst_ip", "src_port", "dst_port", "proto")


def _flow_key(pkt_meta: Dict) -> tuple:
    return tuple(pkt_meta.get(k, "") for k in FLOW_KEYS)


def read_packets(pcap_path: Path) -> List[Dict]:
    """Read all packets from a pcap and return list of per-packet dicts."""
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP, IPv6, ARP, DNS, DNSQR
    packets = rdpcap(str(pcap_path))
    out: List[Dict] = []
    for i, pkt in enumerate(packets):
        meta: Dict = {"index": i, "ts": float(pkt.time)}
        if IP in pkt:
            meta["src_ip"] = pkt[IP].src
            meta["dst_ip"] = pkt[IP].dst
            meta["proto"]  = int(pkt[IP].proto)
            meta["ip_version"] = 4
        elif IPv6 in pkt:
            meta["src_ip"] = pkt[IPv6].src
            meta["dst_ip"] = pkt[IPv6].dst
            meta["proto"]  = int(pkt[IPv6].nh)
            meta["ip_version"] = 6
        elif ARP in pkt:
            meta["src_ip"] = "arp"
            meta["dst_ip"] = "arp"
            meta["proto"]  = "arp"
            meta["ip_version"] = 0
        else:
            meta["src_ip"] = ""
            meta["dst_ip"] = ""
            meta["proto"]  = ""
            meta["ip_version"] = 0
        # transport layer
        if TCP in pkt:
            meta["src_port"] = int(pkt[TCP].sport)
            meta["dst_port"] = int(pkt[TCP].dport)
            flags = int(pkt[TCP].flags)
            meta["flags"] = {
                "fin": bool(flags & 0x01),
                "syn": bool(flags & 0x02),
                "rst": bool(flags & 0x04),
                "psh": bool(flags & 0x08),
                "ack": bool(flags & 0x10),
                "urg": bool(flags & 0x20),
                "ece": bool(flags & 0x40),
                "cwr": bool(flags & 0x80),
            }
        elif UDP in pkt:
            meta["src_port"] = int(pkt[UDP].sport)
            meta["dst_port"] = int(pkt[UDP].dport)
            meta["flags"] = {}
        else:
            meta["src_port"] = 0
            meta["dst_port"] = 0
            meta["flags"] = {}
        meta["length"] = int(len(pkt))
        # application-layer hints
        meta["is_http"]  = (TCP in pkt and (meta["dst_port"] in (80, 8080) or meta["src_port"] in (80, 8080)))
        meta["is_https"] = (TCP in pkt and meta["dst_port"] == 443)
        meta["is_dns"]   = (UDP in pkt and meta["dst_port"] == 53) or (UDP in pkt and meta["src_port"] == 53)
        meta["is_telnet"] = (TCP in pkt and meta["dst_port"] == 23)
        meta["is_smtp"]   = (TCP in pkt and meta["dst_port"] in (25, 587))
        meta["is_ssh"]    = (TCP in pkt and meta["dst_port"] == 22)
        meta["is_irc"]    = (TCP in pkt and meta["dst_port"] in (6667, 6668))
        meta["is_tcp"]    = TCP in pkt
        meta["is_udp"]    = UDP in pkt
        meta["is_dhcp"]   = (UDP in pkt and meta["dst_port"] == 67) or (UDP in pkt and meta["src_port"] == 67)
        meta["is_arp"]    = ARP in pkt
        meta["is_ipv4"]   = IP in pkt
        meta["is_ipv6"]   = IPv6 in pkt
        meta["is_icmp"]   = ICMP in pkt
        out.append(meta)
    return out


def aggregate_flows(packets: List[Dict]) -> Dict[tuple, List[Dict]]:
    """Group packets into bidirectional flows by 5-tuple (canonicalised)."""
    flows: Dict[tuple, List[Dict]] = defaultdict(list)
    for p in packets:
        # canonical 5-tuple (smallest ip first)
        key_a = (p["src_ip"], p["dst_ip"], p["src_port"], p["dst_port"], p["proto"])
        key_b = (p["dst_ip"], p["src_ip"], p["dst_port"], p["src_port"], p["proto"])
        canon = min(key_a, key_b)
        flows[canon].append(p)
    return flows


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def compute_flow_features(flow_packets: List[Dict]) -> Dict[str, float]:
    """Compute the CICIoT2023-style feature dict for one flow."""
    n = len(flow_packets)
    if n == 0:
        return {col: 0.0 for col in FEATURE_COLUMNS}

    durations = [p["ts"] for p in flow_packets]
    flow_duration = max(durations) - min(durations) if n > 1 else 0.0
    lengths = [p["length"] for p in flow_packets]
    iats = [durations[i+1] - durations[i] for i in range(n - 1)]

    flags = defaultdict(int)
    for p in flow_packets:
        for fname, fval in p.get("flags", {}).items():
            if fval:
                flags[fname] += 1

    proto_count = defaultdict(int)
    for p in flow_packets:
        proto_count[p["proto"]] += 1

    # active/idle (gap between packets > 1 sec => idle)
    active_durations: List[float] = []
    idle_durations: List[float] = []
    if n > 1:
        cur_active = 0.0
        for i in range(n - 1):
            gap = durations[i+1] - durations[i]
            if gap < 1.0:
                cur_active += gap
            else:
                active_durations.append(cur_active)
                idle_durations.append(gap)
                cur_active = 0.0
        active_durations.append(cur_active)
    if not active_durations:
        active_durations = [0.0]
    if not idle_durations:
        idle_durations = [0.0]

    n_tcp = sum(1 for p in flow_packets if p["is_tcp"])
    n_udp = sum(1 for p in flow_packets if p["is_udp"])
    n_icmp = sum(1 for p in flow_packets if p["is_icmp"])
    n_arp = sum(1 for p in flow_packets if p["is_arp"])
    n_ipv4 = sum(1 for p in flow_packets if p["is_ipv4"])
    n_ipv6 = sum(1 for p in flow_packets if p["is_ipv6"])

    rate = n / max(flow_duration, 1e-6)   # packets per second
    srate = rate                            # forward rate (approx)
    drate = rate                            # backward rate (approx)

    # protocol type — most common proto
    proto_type = max(proto_count.items(), key=lambda kv: kv[1])[0] if proto_count else 0
    try:
        proto_type = float(proto_type)
    except (TypeError, ValueError):
        proto_type = 0.0

    features = {
        "flow_duration": float(flow_duration),
        "Header_Length": float(sum(lengths)),
        "Protocol Type": proto_type,
        "Duration": float(flow_duration),
        "Rate": float(rate),
        "Srate": float(srate),
        "Drate": float(drate),
        "fin_flag_count": float(flags.get("fin", 0)),
        "syn_flag_count": float(flags.get("syn", 0)),
        "rst_flag_count": float(flags.get("rst", 0)),
        "psh_flag_count": float(flags.get("psh", 0)),
        "ack_flag_count": float(flags.get("ack", 0)),
        "ece_flag_count": float(flags.get("ece", 0)),
        "cwr_flag_count": float(flags.get("cwr", 0)),
        "ack_count":  float(flags.get("ack", 0)),
        "syn_count":  float(flags.get("syn", 0)),
        "fin_count":  float(flags.get("fin", 0)),
        "urg_count":  float(flags.get("urg", 0)),
        "rst_count":  float(flags.get("rst", 0)),
        "HTTP":  float(sum(1 for p in flow_packets if p["is_http"])),
        "HTTPS": float(sum(1 for p in flow_packets if p["is_https"])),
        "DNS":   float(sum(1 for p in flow_packets if p["is_dns"])),
        "Telnet":float(sum(1 for p in flow_packets if p["is_telnet"])),
        "SMTP":  float(sum(1 for p in flow_packets if p["is_smtp"])),
        "SSH":   float(sum(1 for p in flow_packets if p["is_ssh"])),
        "IRC":   float(sum(1 for p in flow_packets if p["is_irc"])),
        "TCP":   float(n_tcp),
        "UDP":   float(n_udp),
        "DHCP":  float(sum(1 for p in flow_packets if p["is_dhcp"])),
        "ARP":   float(n_arp),
        "IPv4":  float(n_ipv4),
        "IPv6":  float(n_ipv6),
        "ICMP":  float(n_icmp),
        "active_std": _safe_std(active_durations),
        "active_max": float(max(active_durations)),
        "active_min": float(min(active_durations)),
        "idle_std":   _safe_std(idle_durations),
        "idle_max":   float(max(idle_durations)),
        "idle_min":   float(min(idle_durations)),
        "Tot sum":  float(sum(lengths)),
        "Min":      float(min(lengths)),
        "Max":      float(max(lengths)),
        "AVG":      float(np.mean(lengths)),
        "Std":      _safe_std([float(x) for x in lengths]),
        "Tot size": float(sum(lengths)),
        "IAT":      float(np.mean(iats)) if iats else 0.0,
        "Number":   float(n),
        "Magnitue": float(np.mean(lengths)),
        "Radius":   float(np.max(lengths) - np.min(lengths)),
        "Covariance": float(np.cov(lengths, durations[:len(lengths)])[0, 1])
                       if len(lengths) > 1 and np.var(lengths) > 0 else 0.0,
        "Variance": float(np.var(lengths)),
        "Weight":   float(n),
    }
    # ensure all expected columns present
    for c in FEATURE_COLUMNS:
        if c not in features:
            features[c] = 0.0
    return features


def pcap_to_flow_rows(pcap_path: Path) -> pd.DataFrame:
    """Read a pcap, aggregate flows, compute features, return DataFrame."""
    packets = read_packets(pcap_path)
    flows = aggregate_flows(packets)
    rows = []
    for key, pkts in flows.items():
        feat = compute_flow_features(pkts)
        feat["_flow_key"] = "|".join(str(k) for k in key)
        feat["_packet_count"] = len(pkts)
        rows.append(feat)
    df = pd.DataFrame(rows)
    return df


def write_mapping_report(out: Path):
    """Document the CICIoT2023 -> simulation feature mapping (spec section 25)."""
    mapping = {
        "note": (
            "CICIoT2023 uses a flow/window-based statistical feature set "
            "(DOI 10.1016/j.iot.2024.101014). This simulation reproduces a "
            "controlled feature-compatible schema, NOT a byte-for-byte copy "
            "of the dataset authors' extractor. Mapping decisions are listed "
            "below and reviewed manually."
        ),
        "features": [
            {"name": "flow_duration", "source": "max(pkt.ts) - min(pkt.ts)", "notes": "seconds"},
            {"name": "Rate / Srate / Drate", "source": "n_packets / flow_duration",
             "notes": "approximate — full CICIoT2023 separates forward / backward rates"},
            {"name": "*_flag_count / *_count", "source": "TCP flag bit counts over the flow", "notes": "exact"},
            {"name": "HTTP / HTTPS / DNS / ...", "source": "port-based classification", "notes": "approximate"},
            {"name": "active_std / active_max / active_min", "source": "time between adjacent pkts < 1s windows",
             "notes": "approximate definition of 'active'"},
            {"name": "Tot sum / Min / Max / AVG / Std / Variance", "source": "packet length statistics", "notes": "exact"},
            {"name": "IAT", "source": "mean inter-arrival time", "notes": "exact"},
            {"name": "Covariance", "source": "cov(lengths, timestamps)",
             "notes": "approximate — original definition uses direction-aware vectors"},
        ],
        "limitations": [
            "Direction-aware forward/backward features are flattened because we "
            "do not always know which endpoint initiated the flow inside the "
            "isolated Docker network.",
            "Protocol Type is encoded as the IANA proto number (TCP=6, UDP=17, "
            "ICMP=1). Verify the dataset's encoding before deployment."
        ],
    }
    write_json(mapping, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-mapping", action="store_true")
    args = ap.parse_args()
    if args.write_mapping:
        out = project_root() / "reports" / "flow_feature_mapping.json"
        write_mapping_report(out)
        print(f"[flow_extraction] wrote mapping -> {out}")
        return
    df = pcap_to_flow_rows(Path(args.pcap))
    out = args.out or str(project_root() / "simulation" / "traffic" / "flows.csv")
    df.to_csv(out, index=False)
    print(f"[flow_extraction] {len(df)} flows written -> {out}")


if __name__ == "__main__":
    main()
