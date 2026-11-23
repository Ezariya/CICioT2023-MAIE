# notebooks/05_edge_evaluation.ipynb (rendered as Python stub)
#
# Goal (spec sections 26-28):
#   - Edge optimization: full -> reduced -> smaller -> optimised model
#   - Raspberry Pi benchmark: 100 / 1000 / 10000 flows
#   - Measure: avg / p95 latency, CPU, RAM, throughput, accuracy, F1, recall
#   - Final question: how much detection performance do we lose to gain
#     real-time edge inference?
import json
from pathlib import Path
import pandas as pd
from src.utils import project_root

rows = []
bench_dir = project_root() / "reports"
for p in sorted(bench_dir.glob("benchmark_edge_*.json")):
    b = json.loads(p.read_text())
    rows.append({
        "model": b["model"],
        "host": b.get("host", "?"),
        "feature_count": b["feature_count"],
        "model_size_kb": b["model_size_kb"],
        "n_rows": b["n_rows"],
        "avg_ms": b["timing"]["avg_inference_ms"],
        "p95_ms": b["timing"]["p95_inference_ms"],
        "throughput_fps": b["timing"]["throughput_flows_per_sec"],
        "ram_mb": b["ram_mb"],
    })

if rows:
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
else:
    print("Run src.evaluation.benchmark --mode edge first.")

# compare PC vs Pi
if rows:
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["model", "feature_count"], columns="host",
                         values=["avg_ms", "p95_ms", "throughput_fps"], aggfunc="mean")
    print("\n=== PC vs Raspberry Pi ===")
    print(piv.to_string())
