# notebooks/04_model_comparison.ipynb (rendered as Python stub)
#
# Goal (spec sections 9, 15):
#   - Compare Logistic Regression / Decision Tree / RF / XGBoost / LightGBM
#   - Across the THREE ML tasks: binary / family / finegrained
#   - Build the trade-off table (Macro F1, Recall, FPR, Latency, Model size)
import json
from pathlib import Path
import pandas as pd
from src.utils import project_root

rows = []
for task in ["binary", "family", "finegrained"]:
    task_dir = project_root() / "models" / (
        {"binary": "binary", "family": "multiclass", "finegrained": "multiclass"}[task]
    )
    if not task_dir.exists():
        continue
    for sub in sorted(task_dir.iterdir()):
        meta_p = sub / "metadata.json"
        if not meta_p.exists():
            continue
        m = json.loads(meta_p.read_text())
        rows.append({
            "task": task,
            "model": m["model"],
            "macro_f1": m["metrics"].get("f1_macro"),
            "recall_macro": m["metrics"].get("recall_macro"),
            "fpr": m["metrics"].get("fpr") or m["metrics"].get("fpr_macro"),
            "fnr": m["metrics"].get("fnr") or m["metrics"].get("fnr_macro"),
            "latency_ms": m["metrics"].get("avg_inference_ms"),
            "model_size_kb": m["metrics"].get("model_size_kb"),
        })

if not rows:
    print("No trained models found yet. Run src.models.train for all tasks first.")
else:
    df = pd.DataFrame(rows)
    df = df.sort_values(["task", "macro_f1"], ascending=[True, False])
    print(df.to_string(index=False))
    out = project_root() / "reports" / "model_comparison_all.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
