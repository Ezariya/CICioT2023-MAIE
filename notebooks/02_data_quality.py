# notebooks/02_data_quality.ipynb (rendered as Python stub)
#
# Goal (spec sections 6, 8):
#   - schema validation: expected features, label, dtype, missing, NaN, inf
#   - duplicate rows
#   - unexpected labels
#   - data leakage audit:
#       * label, attack category, timestamp, file/source info
#       * anything derived from label
#   - suspiciously predictive single features
# Read the leakage audit produced by src.data.clean
import json
from pathlib import Path
from src.utils import project_root

audit_path = project_root() / "reports" / "leakage_audit.json"
audit = json.loads(audit_path.read_text())
print(f"Initial rows       : {audit['initial_rows']:,}")
print(f"Dropped columns    : {audit['dropped_columns_documented']}")
print(f"Bad-label rows     : {audit['rows_dropped_bad_label']:,}")
print(f"Duplicate rows     : {audit['duplicate_rows_removed']:,}")
print(f"Final rows         : {audit['final_rows']:,}")
print(f"\nLeakage audit decisions:")
for d in audit["leakage_audit"]["drop"]:
    print(f"  DROP   {d['column']:<15} present={d['present_in_data']} — {d['reason']}")
for d in audit["leakage_audit"]["kept_documented"]:
    print(f"  KEEP   {d['column']:<15} present={d['present_in_data']} — {d['reason']}")
if audit["leakage_audit"]["flagged_suspicious"]:
    print(f"\nFlagged as suspiciously predictive (review before deployment):")
    for s in audit["leakage_audit"]["flagged_suspicious"]:
        print(f"  {s['column']}: ratio={s['class_mean_diff_ratio']}  ({s['note']})")
else:
    print("\nNo single-feature leakage flagged.")
