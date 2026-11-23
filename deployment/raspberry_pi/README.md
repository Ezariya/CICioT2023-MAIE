# deployment/raspberry_pi/README.md
#
# Raspberry Pi deployment guide (spec sections 26, 27).
#
# The Pi ONLY runs inference. Training stays on your main PC.

## 1. Hardware

Tested on:
  - Raspberry Pi 4B (4 GB RAM)
  - Raspberry Pi 3B+ (1 GB RAM, slower)
  - 32 GB A2 microSD
  - USB Ethernet adapter (for monitor mode if needed)

## 2. OS + dependencies

Flash Raspberry Pi OS Lite (64-bit, bookworm) with the Raspberry Pi Imager.

On the Pi:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev tcpdump tshark libpcap0.8 build-essential
python3 -m venv ~/idsenv
source ~/idsenv/bin/activate
pip install --upgrade pip
pip install scapy pandas numpy scikit-learn joblib xgboost lightgbm pyyaml psutil streamlit
```

## 3. Copy the trained model from your PC

On the PC (NOT the Pi):

```bash
# Train and export a model first
python -m src.models.train --task binary --model XGBoost --dataset D
python -m src.inference.predictor --model-dir models/binary/XGBoost --export
```

Then copy to the Pi:

```bash
scp -r models/binary/XGBoost pi@<pi-ip>:~/ciciot2023-ids/models/binary/
scp -r config pi@<pi-ip>:~/ciciot2023-ids/
scp -r src pi@<pi-ip>:~/ciciot2023-ids/
```

The Pi never runs `src.data.ingest`, `src.data.clean`, `src.data.sample`,
`src.data.split`, or `src.models.train`.

## 4. Run the live IDS on the Pi

```bash
cd ~/ciciot2023-ids
source ~/idsenv/bin/activate
export PYTHONPATH=$(pwd)
sudo $(which python) -m src.simulation.simulator --live --duration 60 \
    --model-dir models/binary/XGBoost --interface eth0
```

`sudo` is needed because raw packet capture requires CAP_NET_RAW.

## 5. Edge benchmark

```bash
for n in 100 1000 10000; do
  python -m src.evaluation.benchmark --mode edge \
      --model-dir models/binary/XGBoost \
      --rows $n
done
```

This writes JSON reports under `reports/benchmark_edge_XGBoost.json` with:

  - avg / p50 / p95 / p99 / max inference_ms
  - throughput_flows_per_sec
  - ram_mb / cpu_percent
  - model_size_kb
  - host=raspberry_pi

Run the same on your PC and compare to answer the question:

> How much detection performance do we lose to obtain a model capable of
> real-time edge inference?

## 6. systemd service (optional)

`sudo cp deployment/raspberry_pi/ids.service /etc/systemd/system/ids.service`
`sudo systemctl daemon-reload`
`sudo systemctl enable --now ids`

## 7. Hard constraints

  - The Pi NEVER trains the model.
  - The Pi NEVER ingests the full CICIoT2023 dataset.
  - The Pi ONLY: loads model.joblib + preprocessing.joblib, captures packets,
    extracts flows, runs inference, logs alerts.
  - The simulation network must be ISOLATED. No attack traffic leaves the lab.
