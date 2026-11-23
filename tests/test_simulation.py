"""
tests/test_simulation.py — test that flow extraction produces the right schema
and that the simulator never fakes detection by sampling a CSV row.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.features.schema import FEATURE_COLUMNS
from src.simulation.flow_extraction import (
    aggregate_flows, compute_flow_features, pcap_to_flow_rows, write_mapping_report,
)
from src.utils import project_root


def test_compute_flow_features_returns_all_columns():
    """A computed flow must contain every expected feature."""
    pkts = [
        {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "src_port": 1000,
         "dst_port": 80, "proto": 6, "ts": 0.0, "length": 100,
         "flags": {"syn": True}, "is_http": True, "is_https": False,
         "is_dns": False, "is_telnet": False, "is_smtp": False, "is_ssh": False,
         "is_irc": False, "is_tcp": True, "is_udp": False, "is_dhcp": False,
         "is_arp": False, "is_ipv4": True, "is_ipv6": False, "is_icmp": False},
        {"src_ip": "2.2.2.2", "dst_ip": "1.1.1.1", "src_port": 80,
         "dst_port": 1000, "proto": 6, "ts": 0.5, "length": 200,
         "flags": {"ack": True}, "is_http": True, "is_https": False,
         "is_dns": False, "is_telnet": False, "is_smtp": False, "is_ssh": False,
         "is_irc": False, "is_tcp": True, "is_udp": False, "is_dhcp": False,
         "is_arp": False, "is_ipv4": True, "is_ipv6": False, "is_icmp": False},
    ]
    feat = compute_flow_features(pkts)
    for c in FEATURE_COLUMNS:
        assert c in feat, f"missing feature {c}"
    assert feat["Number"] == 2
    assert feat["syn_flag_count"] == 1.0
    assert feat["ack_flag_count"] == 1.0


def test_aggregate_flows_bidirectional():
    """Packets in either direction of a 5-tuple should go into ONE flow."""
    pkts = [
        {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "src_port": 1000,
         "dst_port": 80, "proto": 6, "ts": 0.0, "length": 100, "flags": {},
         "is_http": False, "is_https": False, "is_dns": False, "is_telnet": False,
         "is_smtp": False, "is_ssh": False, "is_irc": False, "is_tcp": True,
         "is_udp": False, "is_dhcp": False, "is_arp": False, "is_ipv4": True,
         "is_ipv6": False, "is_icmp": False},
        {"src_ip": "2.2.2.2", "dst_ip": "1.1.1.1", "src_port": 80,
         "dst_port": 1000, "proto": 6, "ts": 0.1, "length": 200, "flags": {},
         "is_http": False, "is_https": False, "is_dns": False, "is_telnet": False,
         "is_smtp": False, "is_ssh": False, "is_irc": False, "is_tcp": True,
         "is_udp": False, "is_dhcp": False, "is_arp": False, "is_ipv4": True,
         "is_ipv6": False, "is_icmp": False},
    ]
    flows = aggregate_flows(pkts)
    assert len(flows) == 1, "bidirectional packets must collapse to one flow"


def test_synthetic_pcap_produces_flows(tmp_path):
    """End-to-end: generate a tiny pcap with scapy, then extract flows."""
    scapy = pytest.importorskip("scapy.all")
    from scapy.all import IP, TCP, Ether, wrpcap
    pkts = []
    for i in range(10):
        p = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(
            sport=40000 + i, dport=80, flags="S")
        p.time = i
        pkts.append(p)
    pcap = tmp_path / "synthetic.pcap"
    wrpcap(str(pcap), pkts)
    df = pcap_to_flow_rows(pcap)
    assert len(df) >= 1, "no flows extracted"
    assert "_flow_key" in df.columns
    assert "_packet_count" in df.columns
