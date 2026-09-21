"""Validate the pinned Bronze layer without connecting to PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.corpus.bronze import validate_bronze_repository  # noqa: E402
from pitchpulse.corpus.training_dataset_validator import (  # noqa: E402
    load_training_corpus_manifest,
)
from pitchpulse.shared.paths import PROJECT_ROOT  # noqa: E402


DEFAULT_CORPUS_MANIFEST = (
    PROJECT_ROOT / "configs" / "datasets" / "vaep-training-corpus-v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the immutable StatsBomb Bronze source and corpus."
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_MANIFEST,
    )
    return parser.parse_args()


def main() -> None:
    corpus = load_training_corpus_manifest(parse_args().corpus_manifest)
    if corpus.bronze is None:
        raise ValueError("a versioned Bronze contract is required")
    source_commit = corpus.source.get("git_commit")
    if not isinstance(source_commit, str):
        raise ValueError("Bronze source requires a pinned git_commit")
    source = validate_bronze_repository(corpus.bronze.data_root, source_commit)
    print(
        json.dumps(
            {
                "status": "passed",
                "corpus_id": corpus.corpus_id,
                "manifest_schema_version": 2,
                "bronze_root": str(source.repository_root),
                "source_commit": source.actual_commit,
                "tracked_tree_clean": source.tracked_tree_clean,
                "competition_seasons": len(corpus.selections),
                "matches": len(corpus.matches),
                "required_families": list(corpus.bronze.required_families),
                "optional_families": list(corpus.bronze.optional_families),
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
