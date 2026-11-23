"""
src/data/inspect.py — lightweight dataset inspection (no Spark needed).

Lists CSV files in data/raw/CICIoT2023, prints file count, total size,
and previews the header of the first file. Useful as a first sanity check
before running the full Spark ingestion in ingest.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.utils import load_config, load_feature_schema, project_root, write_json


def list_csv_files(raw_dir: Path) -> list[Path]:
    """Return every *.csv under raw_dir (recursively)."""
    return sorted(raw_dir.rglob("*.csv"))


def preview_header(first_csv: Path, n: int = 1) -> list[str]:
    """Read the first row of a CSV and return its column names."""
    with first_csv.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header


def file_size_summary(files: list[Path]) -> dict:
    """Compute total size in MB across the given files."""
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "total_gb": round(total_bytes / (1024 * 1024 * 1024), 4),
    }


def main() -> None:
    cfg = load_config()
    schema = load_feature_schema()
    raw_dir = project_root() / cfg["dataset"]["csv_dir"]
    print(f"[inspect] scanning {raw_dir}")
    if not raw_dir.exists():
        print(f"[inspect] directory {raw_dir} does not exist yet.")
        print("         Download the CICIoT2023 CSV release from")
        print("         https://www.unb.ca/cic/datasets/iotdataset-2023.html")
        print("         and unzip into that directory.")
        return

    files = list_csv_files(raw_dir)
    if not files:
        print(f"[inspect] no CSV files found in {raw_dir}")
        return

    summary = file_size_summary(files)
    summary["first_file"] = str(files[0])
    summary["header"] = preview_header(files[0])

    expected_features = [f["name"] for f in schema["features"]]
    summary["expected_feature_count"] = len(expected_features)
    summary["label_column"] = schema["target"]["name"]

    # Compare the file's header to our expected feature schema.
    header = summary["header"]
    expected = expected_features + [schema["target"]["name"]]
    summary["header_matches_expected"] = (set(header) == set(expected))
    summary["missing_in_file"] = sorted(set(expected) - set(header))
    summary["extra_in_file"] = sorted(set(header) - set(expected))

    out_path = project_root() / "data" / "metadata" / "raw_inspection.json"
    write_json(summary, out_path)
    print(f"[inspect] wrote {out_path}")
    print(json_pretty(summary))


def json_pretty(d: dict) -> str:
    import json
    return json.dumps(d, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
