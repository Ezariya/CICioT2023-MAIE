"""
src/simulation/alert.py — minimal alerting system (spec section 29).

  Prediction
      |
      v
  if attack:
      |-- timestamp
      |-- attack class
      |-- confidence
      |-- source flow information
      |-- severity

Example output:
  [ALERT]
  Time: 2026-11-20 19:32:14
  Detection: ATTACK
  Class: DDoS
  Confidence: 0.97
  Severity: HIGH

Detection first. Response (blocking) is an OPTIONAL extension.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import load_config, project_root


@dataclass
class Alert:
    timestamp: str
    detection: str           # "ATTACK" or "BENIGN"
    attack_class: str        # family label for attacks, "" for benign
    confidence: float
    severity: str            # HIGH / MEDIUM / LOW
    flow_key: str
    packet_count: int
    raw: Dict

    def format(self) -> str:
        lines = [
            "[ALERT]" if self.detection == "ATTACK" else "[INFO]",
            f"Time: {self.timestamp}",
            f"Detection: {self.detection}",
        ]
        if self.detection == "ATTACK":
            lines.append(f"Class: {self.attack_class}")
            lines.append(f"Confidence: {self.confidence:.4f}")
            lines.append(f"Severity: {self.severity}")
        lines.append(f"Flow: {self.flow_key}")
        lines.append(f"Packets: {self.packet_count}")
        return "\n".join(lines)


def severity_for(confidence: float, cfg: Dict) -> str:
    thr = cfg["simulation"]["alert"]["severity_thresholds"]
    if confidence >= thr["high"]:
        return "HIGH"
    if confidence >= thr["medium"]:
        return "MEDIUM"
    return "LOW"


class AlertLogger:
    """Append-only alert logger writing to simulation/traffic/alerts.log."""

    def __init__(self, log_path: Path | None = None):
        cfg = load_config()
        if log_path is None:
            log_path = project_root() / cfg["simulation"]["alert"]["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self.alerts: List[Alert] = []
        self._fh = open(log_path, "a", encoding="utf-8")

    def log(self, prediction: Dict, flow_row: Dict) -> Optional[Alert]:
        """Build an alert from a predictor result + flow row."""
        cfg = load_config()
        is_attack = (prediction.get("prediction") == 1) or (
            str(prediction.get("label", "")).lower() not in ("benign", "0")
        )
        sev = severity_for(float(prediction.get("confidence") or 0.0), cfg) if is_attack else "NONE"
        alert = Alert(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            detection="ATTACK" if is_attack else "BENIGN",
            attack_class=str(prediction.get("label", "")) if is_attack else "",
            confidence=float(prediction.get("confidence") or 0.0),
            severity=sev,
            flow_key=str(flow_row.get("_flow_key", "?")),
            packet_count=int(flow_row.get("_packet_count", 0)),
            raw={k: v for k, v in prediction.items() if k != "raw"},
        )
        self.alerts.append(alert)
        # write to file
        self._fh.write(alert.format() + "\n" + "-" * 40 + "\n")
        self._fh.flush()
        return alert if is_attack else None

    def close(self):
        self._fh.close()


def main():
    """Tiny smoke: produce one example alert."""
    cfg = load_config()
    logger = AlertLogger()
    pred = {"prediction": 1, "label": "DDoS", "confidence": 0.97, "inference_ms": 0.3}
    flow = {"_flow_key": "10.0.0.1|10.0.0.2|443|54321|6", "_packet_count": 240}
    a = logger.log(pred, flow)
    print(a.format() if a else "no alert")
    logger.close()


if __name__ == "__main__":
    main()
