"""Validate the pinned Bronze layer without connecting to PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.dataset.bronze import validate_bronze_repository  # noqa: E402
from pitchpulse.dataset.training_dataset_validator import (  # noqa: E402
    TRAINING_DATASET_SCHEMA_VERSION,
    load_training_dataset_manifest,
)
from pitchpulse.shared.paths import PROJECT_ROOT  # noqa: E402


DEFAULT_DATASET_MANIFEST = (
    PROJECT_ROOT / "configs" / "datasets" / "vaep-training-dataset-v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the immutable StatsBomb Bronze source and dataset."
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    return parser.parse_args()


def main() -> None:
    dataset = load_training_dataset_manifest(parse_args().dataset_manifest)
    if dataset.bronze is None:
        raise ValueError("a versioned Bronze contract is required")
    source_commit = dataset.source.get("git_commit")
    if not isinstance(source_commit, str):
        raise ValueError("Bronze source requires a pinned git_commit")
    source = validate_bronze_repository(dataset.bronze.data_root, source_commit)
    print(
        json.dumps(
            {
                "status": "passed",
                "dataset_id": dataset.dataset_id,
                "manifest_schema_version": TRAINING_DATASET_SCHEMA_VERSION,
                "bronze_root": str(source.repository_root),
                "source_commit": source.actual_commit,
                "tracked_tree_clean": source.tracked_tree_clean,
                "competition_seasons": len(dataset.selections),
                "matches": len(dataset.matches),
                "required_families": list(dataset.bronze.required_families),
                "optional_families": list(dataset.bronze.optional_families),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Bronze validation failed: {exc}") from exc
