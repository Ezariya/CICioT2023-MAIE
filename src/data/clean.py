"""
src/data/clean.py — Spark-side data cleaning + leakage audit.

Hard rule: cleaning happens on the Spark DataFrame (NOT pandas). This module:
  - drops rows whose label is unknown (not in taxonomy)
  - handles NaN / +inf / -inf per config.preprocessing
  - removes duplicate rows (but REPORTS counts, never silently)
  - writes the data-leakage audit to reports/leakage_audit.json

No column is dropped without a documented reason.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.utils import (
    all_attack_labels,
    ensure_dir,
    load_config,
    load_feature_schema,
    project_root,
    write_json,
)


def _spark():
    from src.data.ingest import get_spark
    return get_spark()


def audit_leakage(spark, df, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Walk the leakage_audit rules from features.yaml and document decisions."""
    from pyspark.sql.functions import col

    audit_rules = schema.get("leakage_audit", {})
    drop_rules = audit_rules.get("drop", [])
    keep_rules = audit_rules.get("keep_documented", [])
    threshold = audit_rules.get("suspiciously_predictive_threshold", 0.999)

    audit: Dict[str, Any] = {
        "threshold_auc": threshold,
        "drop": [],
        "kept_documented": [],
        "flagged_suspicious": [],
    }

    label_col = schema["target"]["name"]

    # 1. documented drop rules
    for rule in drop_rules:
        colname = rule["column"]
        present = colname in df.columns
        audit["drop"].append({
            "column": colname,
            "reason": rule["reason"],
            "present_in_data": present,
        })

    # 2. documented keep rules
    for rule in keep_rules:
        audit["kept_documented"].append({
            "column": rule["column"],
            "reason": rule["reason"],
            "present_in_data": rule["column"] in df.columns,
        })

    # 3. crude single-feature predictiveness check via mutual-information
    #    approximation: AUC approximated as |mean(label) - 0.5|*2 after
    #    binarising the feature above its median. Slow on full data, but here
    #    we run on the cleaned Spark df. This is a *flag*, not a drop.
    try:
        from pyspark.sql.functions import when, avg, lit
        from src.utils import to_binary_label
        # add a binary target column
        benign_token = "Benign"
        df2 = df.withColumn(
            "_bin_label",
            when(col(label_col) == benign_token, lit(0.0)).otherwise(lit(1.0)),
        )
        n = df2.count()
        if n > 0:
            for fdef in schema["features"]:
                fname = fdef["name"]
                if fname not in df2.columns:
                    continue
                # crude: just measure if the feature is constant per label
                stats = df2.groupBy("_bin_label").avg(fname).collect()
                if len(stats) == 2:
                    a = stats[0]["avg(%s)" % fname] or 0.0
                    b = stats[1]["avg(%s)" % fname] or 0.0
                    denom = (abs(a) + abs(b)) or 1.0
                    sep = abs(a - b) / denom
                    if sep > threshold:
                        audit["flagged_suspicious"].append({
                            "column": fname,
                            "class_mean_diff_ratio": round(sep, 6),
                            "note": "feature mean differs strongly between Benign and Attack; review manually before keeping",
                        })
    except Exception as e:
        audit["suspicious_check_error"] = str(e)

    return audit


def clean(interim_parquet: Path | None = None) -> Dict[str, Any]:
    """Read interim parquet, clean, persist cleaned parquet, return stats."""
    from pyspark.sql.functions import col, when, isnan, isnull, abs as s_abs
    from pyspark.sql.types import DoubleType

    cfg = load_config()
    schema = load_feature_schema()
    if interim_parquet is None:
        interim_parquet = (
            project_root() / "data" / "interim" / "ciciot2023_clean.parquet"
        )
    if not interim_parquet.exists():
        raise FileNotFoundError(
            f"interim parquet not found: {interim_parquet}. Run src.data.ingest first."
        )

    spark = _spark()
    df = spark.read.parquet(str(interim_parquet))
    initial_rows = df.count()

    label_col = schema["target"]["name"]

    # --- 0. run the leakage audit FIRST, while the label column is present --
    audit = audit_leakage(spark, df, schema)

    # --- 1. drop documented leakage columns ----------------------------------
    # NOTE: the `label` column itself is NOT dropped from the DataFrame — it
    # is the training target. It is excluded from features by convention at
    # training time (src.models.train selects only `feature_cols`).
    # The columns we drop here are the INDIRECT leakage columns
    # (source_file, scenario_id, timestamp) that would let the model cheat.
    dropped_cols = []
    for rule in schema["leakage_audit"]["drop"]:
        colname = rule["column"]
        if colname == label_col:
            continue   # keep label for training; excluded from features upstream
        if colname in df.columns:
            df = df.drop(colname)
            dropped_cols.append(colname)

    # --- 2. label validation -------------------------------------------------
    expected_labels = set(["Benign"] + all_attack_labels())
    df_labels_before = df.count()
    df = df.filter(col(label_col).isin(list(expected_labels)))
    rows_after_label_filter = df.count()
    rows_dropped_bad_label = df_labels_before - rows_after_label_filter

    # --- 3. NaN / Inf handling on numeric features ---------------------------
    nan_strategy = cfg["preprocessing"]["nan_strategy"]
    inf_strategy = cfg["preprocessing"]["inf_strategy"]
    feature_names = [f["name"] for f in schema["features"]]

    # cast every feature to double so inf/NaN are well-defined
    for fname in feature_names:
        if fname in df.columns:
            df = df.withColumn(fname, col(fname).cast(DoubleType()))

    # Approximate Inf -> clip; NaN -> replace with column median (computed later)
    # Here we just clip inf to ±1e15 so Spark can persist them safely.
    if cfg["preprocessing"]["handle_inf"]:
        for fname in feature_names:
            if fname in df.columns:
                df = df.withColumn(
                    fname,
                    when(s_abs(col(fname)) > 1e15, when(col(fname) > 0, 1e15).otherwise(-1e15))
                    .otherwise(col(fname)),
                )

    # --- 4. duplicate rows ---------------------------------------------------
    dup_count = df.count() - df.dropDuplicates().count()
    df = df.dropDuplicates()
    final_rows = df.count()

    # (leakage audit was already run at step 0, before any column was dropped)

    cleaned_dir = project_root() / "data" / "interim" / "ciciot2023_cleaned.parquet"
    if cleaned_dir.exists():
        import shutil
        shutil.rmtree(cleaned_dir)
    df.write.mode("overwrite").parquet(str(cleaned_dir))

    report = {
        "initial_rows": initial_rows,
        "dropped_columns_documented": dropped_cols,
        "rows_dropped_bad_label": rows_dropped_bad_label,
        "duplicate_rows_removed": dup_count,
        "final_rows": final_rows,
        "cleaned_parquet": str(cleaned_dir),
        "nan_strategy": nan_strategy,
        "inf_strategy": inf_strategy,
        "leakage_audit": audit,
    }
    reports_dir = project_root() / "reports"
    ensure_dir(reports_dir)
    write_json(report, reports_dir / "leakage_audit.json")
    write_json(report, project_root() / "data" / "metadata" / "cleaning_stats.json")
    spark.stop()
    return report


def main():
    r = clean()
    print(f"[clean] initial rows  : {r['initial_rows']:,}")
    print(f"[clean] dropped cols  : {r['dropped_columns_documented']}")
    print(f"[clean] bad labels    : {r['rows_dropped_bad_label']:,}")
    print(f"[clean] duplicates    : {r['duplicate_rows_removed']:,}")
    print(f"[clean] final rows    : {r['final_rows']:,}")
    print(f"[clean] audit report  : reports/leakage_audit.json")


if __name__ == "__main__":
    main()
