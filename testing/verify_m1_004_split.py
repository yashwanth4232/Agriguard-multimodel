"""M1-004 verification for reproducible dataset split."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import dataset  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_split_artifacts() -> dict[str, object]:
    rows = dataset.load_split_assignments()
    metadata = json.loads(config.SPLIT_METADATA_PATH.read_text(encoding="utf-8"))

    split_counts = Counter(row["split"] for row in rows)
    total = sum(split_counts.values())
    paths = [row["relative_path"] for row in rows]
    duplicate_paths = [p for p in paths if paths.count(p) > 1]

    unit_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        unit_splits[row["unit_id"]].add(row["split"])
    unit_leakage = [unit_id for unit_id, splits in unit_splits.items() if len(splits) > 1]

    per_class = metadata["per_class_split_counts"]
    classes_missing_split = []
    for class_name, counts in per_class.items():
        if counts["train"] == 0 or counts["validation"] == 0 or counts["test"] == 0:
            classes_missing_split.append(class_name)

    return {
        "total_images": total,
        "expected_total": 54305,
        "total_matches_expected": total == 54305,
        "split_counts": dict(split_counts),
        "split_sum_matches_total": sum(split_counts.values()) == total,
        "image_level_duplicate_assignments": len(set(duplicate_paths)),
        "image_level_leakage_pass": len(set(duplicate_paths)) == 0,
        "unit_level_leakage_units": unit_leakage,
        "unit_level_leakage_pass": len(unit_leakage) == 0,
        "duplicate_group_leakage_count": metadata["duplicate_group_leakage_count"],
        "duplicate_group_leakage_pass": metadata["duplicate_group_leakage_count"] == 0,
        "classes_with_missing_split_partition": classes_missing_split,
        "all_classes_represented_pass": len(classes_missing_split) == 0,
    }


def verify_reproducibility() -> dict[str, object]:
    first_hash = file_sha256(config.SPLIT_ASSIGNMENTS_PATH)
    first_rows = dataset.load_split_assignments()

    rows, _ = dataset.create_split_assignments(random_seed=config.RANDOM_SEED)
    regenerated_hash = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode("utf-8")
    ).hexdigest()
    original_hash = hashlib.sha256(
        json.dumps(first_rows, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "assignments_file_hash_stable_on_rerun_logic": regenerated_hash == original_hash,
        "deterministic_pass": regenerated_hash == original_hash,
    }


def main() -> None:
    checks = verify_split_artifacts()
    repro = verify_reproducibility()
    result = {**checks, **repro}
    print(json.dumps(result, indent=2))

    required = [
        checks["total_matches_expected"],
        checks["image_level_leakage_pass"],
        checks["unit_level_leakage_pass"],
        checks["duplicate_group_leakage_pass"],
        checks["all_classes_represented_pass"],
        repro["deterministic_pass"],
    ]
    if not all(required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
