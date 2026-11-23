"""
src/data/ingest.py — PySpark ingestion of the CICIoT2023 CSV corpus.

Hard rule (spec section 4): NEVER load the full 46M-row corpus into a single
pandas DataFrame. The ingestion layer uses Spark to:

  1. discover every CSV file
  2. read them with a single SparkSession (multi-file CSV read)
  3. infer / validate the schema against config/features.yaml
  4. standardise column names (whitespace -> underscore, lower-cased alias)
  5. validate feature dtypes
  6. validate labels against the label_taxonomy in config.yaml
  7. compute dataset statistics and persist them to data/metadata/

Outputs:
  - data/interim/ciciot2023_clean.parquet   (cleaned Spark table)
  - data/metadata/ingestion_stats.json      (rows, files, class counts, ...)
  - data/metadata/schema_report.json        (per-column dtype + missing + inf)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.utils import (
    all_attack_labels,
    ensure_dir,
    load_config,
    load_feature_schema,
    project_root,
    write_json,
)


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------
def get_spark() -> "SparkSession":  # noqa: F821 (type from pyspark)
    """Create and configure a SparkSession for ingestion."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import lit
    except ImportError as e:
        raise RuntimeError(
            "PySpark is required for ingestion. Install with: pip install pyspark"
        ) from e

    cfg = load_config()
    spark_cfg = cfg["spark"]
    builder = (
        SparkSession.builder.appName(spark_cfg["app_name"])
        .master(spark_cfg["master"])
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.executor.memory", spark_cfg["executor_memory"])
        .config("spark.executor.cores", spark_cfg["executor_cores"])
        .config("spark.sql.shuffle.partitions", spark_cfg["shuffle_partitions"])
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def standardise_column(name: str) -> str:
    """Standardise column name: strip, replace spaces with underscores."""
    return name.strip()


def validate_columns(df_columns: List[str], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a report comparing Spark columns with the expected schema."""
    expected_features = [f["name"] for f in schema["features"]]
    label_col = schema["target"]["name"]
    expected_all = expected_features + [label_col]

    spark_cols_set = {standardise_column(c) for c in df_columns}
    expected_set = set(expected_all)

    return {
        "spark_columns": df_columns,
        "expected_columns": expected_all,
        "missing_in_spark": sorted(expected_set - spark_cols_set),
        "extra_in_spark": sorted(spark_cols_set - expected_set),
        "feature_count_expected": len(expected_features),
        "label_column_expected": label_col,
        "schema_matches": (spark_cols_set == expected_set),
    }


def validate_labels(spark, df, label_col: str) -> Dict[str, Any]:
    """Count distinct labels and verify they all belong to the taxonomy."""
    expected_labels = set(["Benign"] + all_attack_labels())
    rows = (
        df.groupBy(label_col).count().orderBy("count", ascending=False).collect()
    )
    label_counts = {r[label_col]: int(r["count"]) for r in rows}
    unknown = sorted(set(label_counts.keys()) - expected_labels)
    return {
        "unique_labels": sorted(label_counts.keys()),
        "label_counts": label_counts,
        "unexpected_labels": unknown,
        "expected_labels": sorted(expected_labels),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_statistics(spark, df, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Per-feature dtype, min, max, mean, stddev, missing %, infinite %."""
    from pyspark.sql.functions import (
        col, count, when, isnan, isnull, abs as s_abs, mean as s_mean,
        stddev as s_std, min as s_min, max as s_max,
    )
    from pyspark.sql.types import NumericType

    n_rows = df.count()
    feature_stats: List[Dict[str, Any]] = []
    for fdef in schema["features"]:
        name = fdef["name"]
        if name not in df.columns:
            feature_stats.append({"name": name, "present": False})
            continue
        c = col(name)
        # cast to double for numeric stats
        num_df = df.select(c.cast("double").alias("v"))
        finite = num_df.filter(~isnan("v") & ~isnull("v"))
        # Inf check: finite after abs > some huge threshold
        inf_count = finite.filter(
            (s_abs("v") > 1e15) | (s_abs("v").isNull())
        ).count()
        null_count = num_df.filter(isnull("v") | isnan("v")).count()
        stats_row = finite.filter(s_abs("v") <= 1e15).agg(
            s_mean("v").alias("mean"),
            s_std("v").alias("std"),
            s_min("v").alias("min"),
            s_max("v").alias("max"),
        ).first()
        stats = {
            "name": name,
            "present": True,
            "dtype": str(df.schema[name].dataType),
            "missing_pct": round(100.0 * null_count / max(n_rows, 1), 4),
            "infinite_pct": round(100.0 * inf_count / max(n_rows, 1), 4),
        }
        if stats_row is not None:
            stats.update({
                "mean":  None if stats_row["mean"] is None else round(float(stats_row["mean"]), 6),
                "std":   None if stats_row["std"] is None else round(float(stats_row["std"]), 6),
                "min":   None if stats_row["min"] is None else round(float(stats_row["min"]), 6),
                "max":   None if stats_row["max"] is None else round(float(stats_row["max"]), 6),
            })
        feature_stats.append(stats)
    return {
        "number_of_rows": n_rows,
        "number_of_features": len(schema["features"]),
        "feature_stats": feature_stats,
    }


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------
def ingest(raw_dir: Path | None = None, sample_for_tests: int | None = None) -> Dict[str, Any]:
    """Run the full ingestion pipeline. Returns the statistics dict."""
    cfg = load_config()
    schema = load_feature_schema()
    if raw_dir is None:
        raw_dir = project_root() / cfg["dataset"]["csv_dir"]
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}. Download the CICIoT2023 CSV release."
        )

    spark = get_spark()

    # Single multi-path CSV read — Spark handles partitions in parallel.
    df = (
        spark.read.option("header", cfg["spark"]["csv_header"])
        .option("inferSchema", cfg["spark"]["csv_infer_schema"])
        .csv([str(p) for p in csv_files])
    )

    # Standardise column names
    for old in df.columns:
        new = standardise_column(old)
        if new != old:
            df = df.withColumnRenamed(old, new)

    label_col = schema["target"]["name"]

    # Schema + label validation
    schema_report = validate_columns(df.columns, schema)
    label_report = validate_labels(spark, df, label_col)
    if not label_report["expected_labels"]:
        raise RuntimeError("no expected labels loaded from config.yaml")

    # For testing, cap rows so the smoke test runs in seconds.
    if sample_for_tests is not None:
        df = df.limit(sample_for_tests)

    # Persist cleaned parquet (interim)
    interim_dir = project_root() / "data" / "interim" / "ciciot2023_clean.parquet"
    if interim_dir.exists():
        # cheap clear — Spark write fails on existing dir by default
        import shutil
        shutil.rmtree(interim_dir)
    df.write.mode("overwrite").parquet(str(interim_dir))

    # Statistics
    stats = compute_statistics(spark, df, schema)

    meta: Dict[str, Any] = {
        "dataset_version": cfg["project"]["version"],
        "config_used": {
            "spark_master": cfg["spark"]["master"],
            "csv_infer_schema": cfg["spark"]["csv_infer_schema"],
            "csv_header": cfg["spark"]["csv_header"],
            "seed": cfg["project"]["seed"],
        },
        "number_of_files": len(csv_files),
        "number_of_rows": stats["number_of_rows"],
        "number_of_features": stats["number_of_features"],
        "unique_labels": label_report["unique_labels"],
        "class_counts": label_report["label_counts"],
        "unexpected_labels": label_report["unexpected_labels"],
        "schema_report": schema_report,
        "feature_stats": stats["feature_stats"],
        "interim_parquet": str(interim_dir),
    }

    md_dir = project_root() / "data" / "metadata"
    ensure_dir(md_dir)
    write_json(meta, md_dir / "ingestion_stats.json")
    write_json(schema_report, md_dir / "schema_report.json")
    spark.stop()
    return meta


def main():
    meta = ingest()
    print(f"[ingest] files    : {meta['number_of_files']}")
    print(f"[ingest] rows     : {meta['number_of_rows']:,}")
    print(f"[ingest] features : {meta['number_of_features']}")
    print(f"[ingest] labels   : {len(meta['unique_labels'])}")
    for lab, n in sorted(meta["class_counts"].items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {lab:<30} {n:>12,}")


if __name__ == "__main__":
    main()
