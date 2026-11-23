# CICIoT2023 IDS — Quick Demo Guide

Your model is trained. Now prove it works.

## TL;DR — 30-second demo (no Docker)

```bash
cd ciciot2023-ids
python demo.py
```

That's it. The script auto-finds your trained model, generates synthetic
traffic (benign + attack), and shows detection results.

## What the demo does

```
  Synthetic Traffic (scapy)          ← benign HTTP flows + SYN flood
         │
         ▼
  Flow Extraction                    ← 52 CICIoT2023-compatible features
         │
         ▼
  ML Inference                       ← your trained XGBoost/RF/LightGBM
         │
         ▼
  Alert (HIGH / MEDIUM / LOW)        ← printed to terminal
```

Three scenarios run automatically:

1. **Benign** — 8 normal HTTP flows (sensor polling). Should all be classified as BENIGN.
2. **Attack** — 300-packet SYN flood in a single flow. Should trigger a HIGH severity ALERT.
3. **Mixed** — benign + attack simultaneously. Should correctly separate them.

## Expected output

```
============================================================
  CICIoT2023 AI-BASED IoT IDS — LIVE DEMO
============================================================

STEP 1: Loading trained model
  [loaded] model=XGBoost  task=binary  features=52  threshold=0.5

STEP 2: Scenario A — Benign IoT Traffic
  [OK]  10.0.0.10|10.0.0.20|50000|80|6   pkts=5   conf=0.9727
  [OK]  10.0.0.10|10.0.0.20|50001|80|6   pkts=5   conf=0.9727
  ... (8 benign flows)

STEP 3: Scenario B — DDoS SYN Flood Attack
  ┌─ [ALERT] ──────────────────────────────────────────────
  │ Time       : 2026-11-23 17:52:39
  │ Detection  : ATTACK
  │ Class      : Attack
  │ Confidence : 0.9983
  │ Flow       : 10.0.0.20|10.0.0.99|80|40000|6
  │ Packets    : 300
  │ Severity   : HIGH
  └──────────────────────────────────────────────────────────

============================================================
  DEMO SUMMARY
============================================================
  Total flows analyzed : 15
  Benign detected      : 13
  Attacks detected     : 2
  Avg inference time   : 0.637 ms/flow
  Throughput           : 1,570 flows/sec
============================================================

⚠ ATTACKS DETECTED — IDS is working!
```

---

## Option 1: Terminal demo (fastest)

```bash
# Auto-finds model in models/binary/XGBoost/
python demo.py

# Or specify a model
python demo.py --model-dir models/binary/XGBoost

# Run just one scenario
python demo.py --scenarios attack
```

**Requirements**: `pip install scapy scikit-learn xgboost joblib pyyaml pandas numpy`

---

## Option 2: Web dashboard (visual)

```bash
streamlit run demo_dashboard.py
```

Then open `http://localhost:8501` in your browser.

The dashboard shows:
- Model metrics in the sidebar (F1, FPR, FNR)
- Buttons to run each scenario
- Real-time alert feed with severity badges
- Flow detection table with color-coded predictions
- Attack distribution chart
- Architecture diagram

**Requirements**: `pip install streamlit plotly scapy scikit-learn xgboost joblib pyyaml pandas numpy`

---

## Option 3: Docker (containerized)

### Quickest: Docker Compose (Cross-platform)

```bash
# Terminal demo
docker compose -f docker-compose.demo.yml up demo

# Web dashboard (recommended)
docker compose -f docker-compose.demo.yml --profile dashboard up dashboard
```

Then open `http://localhost:8501`.

---

### Or via `docker run`:

**Build image first:**
```bash
docker build -t ciciot2023-ids-demo -f Dockerfile.demo .
```

**Run Web Dashboard:**

- **On Windows (PowerShell):**
```powershell
docker run --rm -p 8501:8501 -v "${PWD}/models:/app/models:ro" ciciot2023-ids-demo streamlit run demo_dashboard.py --server.port=8501 --server.address=0.0.0.0
```

- **On Linux / macOS (bash):**
```bash
docker run --rm -p 8501:8501 -v "$(pwd)/models:/app/models:ro" ciciot2023-ids-demo streamlit run demo_dashboard.py --server.port=8501 --server.address=0.0.0.0
```

**Run Terminal Demo:**

- **On Windows (PowerShell):**
```powershell
docker run --rm -v "${PWD}/models:/app/models:ro" ciciot2023-ids-demo
```

- **On Linux / macOS (bash):**
```bash
docker run --rm -v "$(pwd)/models:/app/models:ro" ciciot2023-ids-demo
```

Then open `http://localhost:8501`.

---

## On Google Colab

If your models are on Colab, run the demo there directly:

```python
# In a Colab cell:
!pip install scapy streamlit

# Terminal demo
!python demo.py --model-dir models/binary/XGBoost

# Dashboard (use Colab's port forwarding or ngrok)
!streamlit run demo_dashboard.py --server.port=8501 &>/dev/null&
```

---

## What to show in your presentation

1. **Run the terminal demo** — `python demo.py`
   - Shows the model loading
   - Shows benign traffic being classified as OK
   - Shows the attack being detected with an ALERT
   - Shows the summary with throughput numbers

2. **Point out the key metrics**:
   - Confidence: 0.9983 for the attack (nearly certain)
   - Inference time: <1 ms per flow
   - Throughput: ~1,500+ flows/sec

3. **Explain the chain**:
   - "The traffic is synthetic, but the detection chain is real"
   - "Packets are grouped into flows, 52 statistical features are extracted"
   - "The trained XGBoost model classifies each flow"
   - "Attacks trigger alerts with confidence and severity"

4. **Optionally show the dashboard** for a more visual presentation.

---

## Files

| File | Purpose |
|------|---------|
| `demo.py` | Self-contained terminal demo (no Docker needed) |
| `demo_dashboard.py` | Streamlit web dashboard |
| `Dockerfile.demo` | Single-container Docker image |
| `docker-compose.demo.yml` | Docker Compose for demo + dashboard |
