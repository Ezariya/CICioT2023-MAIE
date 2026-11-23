"""
src/data/split.py — train / validation / test split.

Hard rule (spec section 13):
  Fit preprocessing ONLY on TRAIN. Never calculate normalization statistics
  on the complete dataset before splitting.

We do the split in Spark (stratified on the label) and write three parquet
files per dataset + task. The ML training stage loads TRAIN, fits any
scaler/encoder on TRAIN only, then transforms VAL and TEST.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from src.utils import (
    ensure_dir,
    load_config,
    load_feature_schema,
    project_root,
    to_binary_label,
    to_family_label,
    write_json,
)


def _spark():
    from src.data.ingest import get_spark
    return get_spark()


def _add_task_label(df, label_col: str, task: str):
    from pyspark.sql.functions import udf, lit, when, col
    from pyspark.sql.types import IntegerType, StringType

    if task == "binary":
        udf_bin = udf(lambda lab: to_binary_label(lab), IntegerType())
        return df.withColumn("y", udf_bin(col(label_col)))
    if task == "family":
        udf_fam = udf(lambda lab: to_family_label(lab), StringType())
        return df.withColumn("y", udf_fam(col(label_col)))
    if task == "finegrained":
        return df.withColumn("y", col(label_col))
    raise ValueError(f"unknown task: {task}")


def stratified_split(spark, df, label_col: str, fractions: Tuple[float, float, float],
                     seed: int = 42):
    """Stratified split into train / val / test using Spark sampleBy."""
    from pyspark.sql.functions import col
    labels = [r[label_col] for r in df.select(label_col).distinct().collect()]
    train_frac, val_frac, test_frac = fractions
    # sampleBy wants per-label fractions
    train_map = {l: train_frac for l in labels}
    val_map   = {l: val_frac   for l in labels}
    test_map  = {l: test_frac  for l in labels}
    train_df = df.sampleBy(label_col, train_map, seed=seed)
    val_df   = df.sampleBy(label_col, val_map,   seed=seed + 1)
    test_df  = df.sampleBy(label_col, test_map,  seed=seed + 2)
    return train_df, val_df, test_df


def split_dataset(dataset_letter: str, task: str) -> Dict:
    cfg = load_config()
    schema = load_feature_schema()
    label_col = schema["target"]["name"]
    split_cfg = cfg["split"]
    seed = split_cfg["seed"]

    sample_dir = (
        project_root() / "data" / "processed"
        / {"A": "binary", "B": "binary", "C": "multiclass", "D": "edge"}[dataset_letter.upper()]
        / f"sampled_{dataset_letter.lower()}.parquet"
    )
    if not sample_dir.exists():
        raise FileNotFoundError(
            f"sampled parquet missing: {sample_dir}. Run src.data.sample {dataset_letter} first."
        )
    spark = _spark()
    df = spark.read.parquet(str(sample_dir))
    df = _add_task_label(df, label_col, task)

    # Stratify on the task label `y` (binary/family/finegrained)
    train_df, val_df, test_df = stratified_split(
        spark, df, "y",
        (split_cfg["train"], split_cfg["val"], split_cfg["test"]),
        seed=seed,
    )

    task_dir = sample_dir.parent / f"{task}_split"
    ensure_dir(task_dir)
    for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = task_dir / f"{name}.parquet"
        if out.exists():
            import shutil
            shutil.rmtree(out)
        sub.write.mode("overwrite").parquet(str(out))

    counts = {
        "train": train_df.count(),
        "val":   val_df.count(),
        "test":  test_df.count(),
    }
    log = {
        "dataset": dataset_letter.upper(),
        "task": task,
        "seed": seed,
        "split_fractions": {
            "train": split_cfg["train"],
            "val": split_cfg["val"],
            "test": split_cfg["test"],
        },
        "row_counts": counts,
        "fit_preprocessing_on": "train",
        "output_dir": str(task_dir),
    }
    write_json(
        log,
        project_root() / "data" / "metadata" / f"split_{dataset_letter.upper()}_{task}.json",
    )
    spark.stop()
    return log


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--task", required=True, choices=["binary", "family", "finegrained"])
    args = ap.parse_args()
    log = split_dataset(args.dataset, args.task)
    print(f"[split] {log['dataset']} / {log['task']}: {log['row_counts']}")


if __name__ == "__main__":
    main()
