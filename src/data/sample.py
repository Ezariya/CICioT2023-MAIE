"""
src/data/sample.py — reproducible stratified sampling.

Hard rule (spec section 4): NEVER use pandas to sample 46M rows. We do it in
Spark. Produces the four datasets A / B / C / D defined in config.yaml:

  A — original distribution sample   (~2,000,000)
  B — balanced binary                (500k Benign + 500k Attack)
  C — stratified 8-class             (~80k per family)
  D — edge-lightweight               (~20k per class)

Every sample has a reproducible seed and the per-class row counts are recorded
in data/metadata/sampling_log.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.utils import (
    ensure_dir,
    load_config,
    project_root,
    to_binary_label,
    to_family_label,
    write_json,
)


def _spark():
    from src.data.ingest import get_spark
    return get_spark()


def _sample_by_label(spark, df, label_col: str, rows_per_label: Dict[str, int], seed: int):
    """Sample `rows_per_label[label]` rows for each label value."""
    from pyspark.sql.functions import col
    parts = []
    counts: Dict[str, int] = {}
    for label, n in rows_per_label.items():
        sub = df.filter(col(label_col) == label)
        avail = sub.count()
        take = min(n, avail)
        if take == 0:
            continue
        frac = take / avail if avail > 0 else 0.0
        # Use sampleBy with a fraction, then cap with limit to get exact count.
        sampled = sub.sample(withReplacement=False, fraction=frac, seed=seed).limit(take)
        parts.append(sampled)
        counts[label] = take
    if not parts:
        raise RuntimeError("no rows sampled — check label filters")
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    return out, counts


def sample_dataset_a(spark, df, label_col: str, cfg) -> Dict:
    """Dataset A — original distribution, capped per class."""
    spec = cfg["sampling"]["datasets"]["A_original_distribution"]
    rpc = spec["rows_per_class"]
    seed = cfg["sampling"]["seed"]
    # Discover all available labels first
    labels = [r[label_col] for r in df.select(label_col).distinct().collect()]
    rows_per_label = {lab: rpc for lab in labels}
    out, counts = _sample_by_label(spark, df, label_col, rows_per_label, seed)
    return {"df": out, "counts": counts, "purpose": spec["purpose"], "seed": seed}


def sample_dataset_b(spark, df, label_col: str, cfg) -> Dict:
    """Dataset B — balanced binary."""
    from pyspark.sql.functions import col, when, lit
    spec = cfg["sampling"]["datasets"]["B_balanced_binary"]
    benign_n = spec["benign_rows"]
    attack_n = spec["attack_rows"]
    seed = cfg["sampling"]["seed"]

    benign = df.filter(col(label_col) == "Benign").sample(
        withReplacement=False, fraction=1.0, seed=seed
    ).limit(benign_n)
    attacks = df.filter(col(label_col) != "Benign").sample(
        withReplacement=False, fraction=1.0, seed=seed
    ).limit(attack_n)
    out = benign.unionByName(attacks)
    counts = {"Benign": benign_n, "Attack": attack_n}
    return {"df": out, "counts": counts, "purpose": spec["purpose"], "seed": seed}


def sample_dataset_c(spark, df, label_col: str, cfg) -> Dict:
    """Dataset C — stratified per 8-family (Benign + 7 attack families)."""
    from pyspark.sql.functions import udf, col, lit
    from pyspark.sql.types import StringType

    spec = cfg["sampling"]["datasets"]["C_stratified_multiclass"]
    rpc = spec["rows_per_family"]
    seed = cfg["sampling"]["seed"]

    from src.utils import to_family_label
    fam_udf = udf(lambda lab: to_family_label(lab), StringType())
    df2 = df.withColumn("family", fam_udf(col(label_col)))

    families = [r["family"] for r in df2.select("family").distinct().collect()]
    rows_per_fam = {fam: rpc for fam in families}
    out, counts = _sample_by_label(spark, df2, "family", rows_per_fam, seed)
    return {
        "df": out.drop("family"),
        "counts": counts,
        "purpose": spec["purpose"],
        "seed": seed,
        "extra_column_dropped": "family",
    }


def sample_dataset_d(spark, df, label_col: str, cfg) -> Dict:
    """Dataset D — edge-lightweight per individual attack label."""
    spec = cfg["sampling"]["datasets"]["D_edge_lightweight"]
    rpc = spec["rows_per_class"]
    seed = cfg["sampling"]["seed"]
    labels = [r[label_col] for r in df.select(label_col).distinct().collect()]
    rows_per_label = {lab: rpc for lab in labels}
    out, counts = _sample_by_label(spark, df, label_col, rows_per_label, seed)
    return {"df": out, "counts": counts, "purpose": spec["purpose"], "seed": seed}


_SAMPLERS = {
    "A": sample_dataset_a,
    "B": sample_dataset_b,
    "C": sample_dataset_c,
    "D": sample_dataset_d,
}


def sample(dataset_letter: str, cleaned_parquet: Path | None = None) -> Dict:
    from src.utils import load_feature_schema
    cfg = load_config()
    schema = load_feature_schema()
    label_col = schema["target"]["name"]
    if cleaned_parquet is None:
        cleaned_parquet = (
            project_root() / "data" / "interim" / "ciciot2023_cleaned.parquet"
        )
    if not cleaned_parquet.exists():
        raise FileNotFoundError(
            f"cleaned parquet missing: {cleaned_parquet}. Run src.data.clean first."
        )
    spark = _spark()
    df = spark.read.parquet(str(cleaned_parquet))
    sampler = _SAMPLERS[dataset_letter.upper()]
    result = sampler(spark, df, label_col, cfg)

    out_dir = (
        project_root() / "data" / "processed"
        / {"A": "binary", "B": "binary", "C": "multiclass", "D": "edge"}[dataset_letter.upper()]
        / f"sampled_{dataset_letter.lower()}.parquet"
    )
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    result["df"].write.mode("overwrite").parquet(str(out_dir))
    log_entry = {
        "dataset": dataset_letter.upper(),
        "purpose": result["purpose"],
        "seed": result["seed"],
        "per_class_rows": result["counts"],
        "total_rows": sum(result["counts"].values()),
        "output": str(out_dir),
    }
    write_json(
        log_entry,
        project_root() / "data" / "metadata" / f"sampling_{dataset_letter.upper()}.json",
    )
    spark.stop()
    return log_entry


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="A", choices=["A", "B", "C", "D", "all"])
    args = ap.parse_args()
    letters = ["A", "B", "C", "D"] if args.dataset == "all" else [args.dataset]
    for L in letters:
        log = sample(L)
        print(f"[sample {L}] {log['purpose']}")
        for cls, n in sorted(log["per_class_rows"].items()):
            print(f"   {cls:<30} {n:>10,}")
        print(f"   -> {log['output']}")


if __name__ == "__main__":
    main()
