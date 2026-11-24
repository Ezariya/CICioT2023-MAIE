"""
src/simulation/traffic_generator.py — Synthetic IoT Traffic & Attack Generator.

Generates realistic, reproducible Scapy packets for:
  - Benign IoT Telemetry (HTTP REST polling, MQTT broker publish, DNS query)
  - DDoS Attacks (TCP SYN Flood, UDP Flood, ICMP Flood)
  - Reconnaissance (TCP SYN Port Scan, Host Discovery)
  - IoT Botnet Activity (Mirai Telnet Brute Force + Flood)
  - Mixed Fleet Telemetry (Concurrent multi-host benign + attack)
"""
from __future__ import annotations

import time
from typing import List, Optional
from scapy.all import Ether, IP, TCP, UDP, ICMP, Raw, DNS, DNSQR


def make_packet(src: str, dst: str, sport: int, dport: int, flags: str, ts: float, payload=None):
    """Build a single scapy TCP packet."""
    if payload:
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags) / Raw(load=payload)
    else:
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    p.time = ts
    return p


def gen_benign_http(base_time: float, n_flows: int = 8, src_ip: str = "10.0.0.10", dst_ip: str = "10.0.0.20") -> List:
    """Generate benign HTTP-like flows: SYN -> SYN-ACK -> ACK -> GET -> ACK."""
    pkts = []
    for f in range(n_flows):
        sport = 50000 + f
        t = base_time + f * 0.3
        pkts.append(make_packet(src_ip, dst_ip, sport, 80, "S", t))
        pkts.append(make_packet(dst_ip, src_ip, 80, sport, "SA", t + 0.01))
        pkts.append(make_packet(src_ip, dst_ip, sport, 80, "A", t + 0.02))
        pkts.append(make_packet(src_ip, dst_ip, sport, 80, "PA", t + 0.03,
                                payload=f"GET /sensor/{f} HTTP/1.1\r\nHost: iot.local\r\n\r\n"))
        pkts.append(make_packet(dst_ip, src_ip, 80, sport, "A", t + 0.04))
    return pkts


def gen_benign_mqtt(base_time: float, n_messages: int = 5, src_ip: str = "10.0.0.12", broker_ip: str = "10.0.0.20") -> List:
    """Generate benign MQTT IoT sensor telemetry over TCP port 1883."""
    pkts = []
    sport = 49152
    t = base_time
    pkts.append(make_packet(src_ip, broker_ip, sport, 1883, "S", t))
    pkts.append(make_packet(broker_ip, src_ip, 1883, sport, "SA", t + 0.01))
    pkts.append(make_packet(src_ip, broker_ip, sport, 1883, "A", t + 0.02))
    pkts.append(make_packet(src_ip, broker_ip, sport, 1883, "PA", t + 0.03, payload="CONNECT client_id=iot_temp_sensor"))
    pkts.append(make_packet(broker_ip, src_ip, 1883, sport, "A", t + 0.04))
    for i in range(n_messages):
        t_msg = t + 0.1 + i * 0.2
        pkts.append(make_packet(src_ip, broker_ip, sport, 1883, "PA", t_msg, payload=f"PUBLISH topic=sensors/temp val=24.{i}"))
        pkts.append(make_packet(broker_ip, src_ip, 1883, sport, "A", t_msg + 0.02))
    return pkts


def gen_syn_flood(base_time: float, n_packets: int = 300, src_ip: str = "10.0.0.99", dst_ip: str = "10.0.0.20", target_port: int = 80) -> List:
    """High-rate DDoS TCP SYN flood targeting a single port."""
    pkts = []
    for i in range(n_packets):
        pkts.append(make_packet(src_ip, dst_ip, 40000, target_port, "S", base_time + i * 0.0001))
    return pkts


def gen_udp_flood(base_time: float, n_packets: int = 300, src_ip: str = "10.0.0.99", dst_ip: str = "10.0.0.20", target_port: int = 1234) -> List:
    """High-rate DDoS UDP packet flood overwhelming network bandwidth."""
    pkts = []
    for i in range(n_packets):
        p = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=40000, dport=target_port)
        p.time = base_time + i * 0.0001
        pkts.append(p)
    return pkts


def gen_port_scan(base_time: float, n_ports: int = 60, src_ip: str = "10.0.0.99", dst_ip: str = "10.0.0.20") -> List:
    """Reconnaissance TCP SYN port scan across many different ports."""
    pkts = []
    for i in range(n_ports):
        pkts.append(make_packet(src_ip, dst_ip, 40000, 1 + i, "S", base_time + i * 0.01))
    return pkts


def gen_mirai_botnet(base_time: float, n_packets: int = 250, src_ip: str = "10.0.0.99", dst_ip: str = "10.0.0.20") -> List:
    """Mirai IoT Botnet simulation: Telnet brute force session followed by rapid flood."""
    pkts = []
    # Phase 1: Telnet brute force handshake
    pkts.append(make_packet(src_ip, dst_ip, 41000, 23, "S", base_time))
    pkts.append(make_packet(dst_ip, src_ip, 23, 41000, "SA", base_time + 0.01))
    pkts.append(make_packet(src_ip, dst_ip, 41000, 23, "A", base_time + 0.02))
    pkts.append(make_packet(src_ip, dst_ip, 41000, 23, "PA", base_time + 0.03, payload="root\r\nxc3511\r\n"))
    # Phase 2: High rate attack flood
    for i in range(n_packets):
        pkts.append(make_packet(src_ip, dst_ip, 42000, 80, "S", base_time + 0.1 + i * 0.0001))
    return pkts


def gen_mixed_fleet(base_time: float) -> List:
    """Mixed Enterprise IoT environment: benign sensor flows + concurrent attacker."""
    benign = gen_benign_http(base_time, n_flows=5)
    attack = gen_syn_flood(base_time + 0.5, n_packets=200)
    return benign + attack
