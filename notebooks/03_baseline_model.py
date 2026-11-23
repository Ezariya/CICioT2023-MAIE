# notebooks/03_baseline_model.ipynb (rendered as Python stub)
#
# Goal (spec section 14):
#   - Logistic Regression baseline
#   - Decision Tree baseline
#   - Random Forest baseline
#   - Train on Dataset B (balanced binary) for the binary task.
#
# Run from the project root:
#   python -m src.models.train --task binary --model LogisticRegression,DecisionTree,RandomForest --dataset B
import json
from pathlib import Path
import pandas as pd
from src.utils import project_root

comparison_csv = project_root() / "reports" / "comparison_binary_B.csv"
if comparison_csv.exists():
    df = pd.read_csv(comparison_csv)
    print("=== Baseline comparison (Dataset B, binary) ===")
    print(df.to_string(index=False))
else:
    print(f"Run src.models.train first; {comparison_csv} not found yet.")

# Read metadata for each model
for model_name in ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost", "LightGBM"]:
    p = project_root() / "models" / "binary" / model_name / "metadata.json"
    if p.exists():
        m = json.loads(p.read_text())
        print(f"\n{model_name}:")
        print(f"  macro_f1   : {m['metrics']['f1_macro']}")
        print(f"  recall     : {m['metrics']['recall_macro']}")
        print(f"  FPR / FNR  : {m['metrics']['fpr']} / {m['metrics']['fnr']}")
        print(f"  model_size : {m['metrics']['model_size_kb']} KB")
        if "threshold_sweep" in m["metrics"]:
            print("  threshold sweep:")
            for t, vals in m["metrics"]["threshold_sweep"].items():
                print(f"    t={t}: F1={vals['f1_macro']}  recall_attack={vals['recall_attack']}  FPR={vals['fpr']}  FNR={vals['fnr']}")
