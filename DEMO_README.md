# 🛡️ CICIoT2023 AI-Powered IoT IDS — Live Demonstration Guide

A presentation-grade, cross-platform demonstration and evaluation platform for AI/ML Intrusion Detection Systems trained on the **CICIoT2023** benchmark.

---

## ⚡ 1-Click Launch (Easiest for Teammates)

The project is completely self-contained. Any teammate on any OS can launch the SOC dashboard with a single command or click:

| Operating System | How to Launch |
| :--- | :--- |
| **Windows (GUI / Double-click)** | Double-click `run_demo.bat` |
| **Windows (PowerShell)** | `.\run_demo.ps1` or `docker compose up` |
| **macOS / Linux / WSL** | `bash run_demo.sh` or `docker compose up` |

Once started, open **[http://localhost:8501](http://localhost:8501)** in your browser.

To stop the dashboard later:
```bash
docker compose down
```

---

## 🖥️ What the Interactive Dashboard Offers

The dashboard is structured into 4 tabs:

### 1. ⚡ Live Attack Simulation
Craft raw packets in memory with Scapy, extract 52 flow statistics in real-time, and run live model inference:
- **🌐 Benign HTTP**: Normal REST temperature & telemetry sensor polling.
- **📡 Benign MQTT**: IoT sensor reporting over port 1883.
- **⚠️ DDoS SYN Flood**: Aggressive 300-packet TCP SYN flood (~10,000 pps).
- **🌊 DDoS UDP Flood**: High-bandwidth UDP blast overwhelming IoT gateways.
- **🔍 Recon Port Scan**: Horizontal TCP probe sweeping common IoT ports (21, 22, 23, 80, 443, 1883).
- **🤖 Mirai IoT Botnet**: Telnet brute-force connection sequence followed by attack burst.
- **🔀 Mixed Multi-Host Fleet**: Realistic multi-device isolation (5 benign sensors + live attacker).
- **🛠️ Custom Injection Studio**: Select attack type, source IP, target IP, and packet volume.
- **🔍 Deep Feature Inspector**: Click any flow to view its 52 features.

### 2. 🧪 Dataset Benchmark Replay
- Evaluates real test flows directly from the official **CICIoT2023** test dataset.
- Filter by attack category (`DDoS`, `DoS`, `Mirai`, `Recon`, `Web-based`, `BruteForce`, `Spoofing`, `Benign`).
- Live confusion comparison: Ground Truth vs Predicted Label, Accuracy, Confidence.

### 3. 📁 PCAP File Analyzer
- Drag-and-drop any `.pcap` or `.pcapng` file (from Wireshark or tcpdump).
- Extracts all bidirectional 5-tuple flows live and provides security verdicts.

### 4. 📊 Model Analytics & Explainability
- Switch between multiple models (`RandomForest`, `XGBoost`, `LightGBM`, `LogisticRegression`, `family_XGBoost`).
- View Accuracy, Macro-F1, False Positive Rate (FPR), False Negative Rate (FNR).
- Top 10 feature importance decision drivers chart.

---

## 💻 Option 2: Terminal Demonstration (CLI)

Fast console evaluation with formatted alerts and automatic executive HTML report generation:

```bash
# Auto-detects best model and runs all 6 scenarios
python demo.py

# Run with real dataset benchmark evaluation
python demo.py --benchmark

# Test a specific model
python demo.py --model-dir models/binary/RandomForest
```

### Outputs Generated:
- **`demo_results.json`**: Structured JSON log of all analyzed flows.
- **`demo_report.html`**: Self-contained visual HTML report (open in any browser).

---

## 🐳 Docker Deployment Options

### Standard `docker run` (Self-Contained)
Because the image is completely self-contained, no volume mounts are required:

```bash
# 1. Build the image (only needed once)
docker build -t ciciot2023-ids-demo -f Dockerfile.demo .

# 2. Run SOC Web Dashboard
docker run --rm -p 8501:8501 ciciot2023-ids-demo

# 3. Run Terminal Demo
docker run --rm ciciot2023-ids-demo python demo.py --benchmark
```

### With Model Volume Mounting (Optional)
If you train new models and want to mount them without rebuilding:

- **Windows PowerShell:**
```powershell
docker run --rm -p 8501:8501 -v "${PWD}/models:/app/models:ro" ciciot2023-ids-demo
```

- **Linux / macOS (bash):**
```bash
docker run --rm -p 8501:8501 -v "$(pwd)/models:/app/models:ro" ciciot2023-ids-demo
```
