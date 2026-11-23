#!/usr/bin/env bash
# deployment/raspberry_pi/install.sh
#
# Run ON the Raspberry Pi to set up the edge IDS runtime.
# This does NOT install training dependencies (no Spark, no XGBoost build).
set -euo pipefail

echo "[install] updating OS packages"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-dev tcpdump tshark libpcap0.8 \
                       build-essential git

VENV=~/idsenv
if [ ! -d "$VENV" ]; then
    echo "[install] creating virtualenv at $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel
pip install scapy==2.5.0 \
            pandas==2.1.* \
            numpy==1.26.* \
            scikit-learn==1.3.* \
            xgboost==2.0.* \
            lightgbm==4.1.* \
            joblib==1.3.* \
            pyyaml==6.* \
            psutil==5.9.* \
            streamlit==1.29.*

echo "[install] installing systemd unit"
REPO_DIR="$(pwd)"
# Patch the unit file with this checkout path
UNIT=/etc/systemd/system/ids.service
if [ -f deployment/raspberry_pi/ids.service ]; then
    sudo cp deployment/raspberry_pi/ids.service "$UNIT"
    sudo sed -i "s|/home/pi/ciciot2023-ids|${REPO_DIR}|g" "$UNIT"
    sudo sed -i "s|/home/pi/idsenv|${VENV}|g" "$UNIT"
    sudo systemctl daemon-reload
    echo "[install] systemd unit installed. Enable with:"
    echo "   sudo systemctl enable --now ids"
fi
echo "[install] done."
