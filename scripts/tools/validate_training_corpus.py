"""Validate the pinned VAEP training corpus without touching PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.ml.corpus import load_training_corpus_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify all metadata hashes and source files in a VAEP corpus."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=(
            PROJECT_ROOT
            / "configs"
            / "datasets"
            / "vaep-training-corpus-v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    corpus = load_training_corpus_manifest(parse_args().manifest)
    print(
        json.dumps(
            {
                "valid": True,
                "corpus_id": corpus.corpus_id,
                "manifest_sha256": corpus.manifest_sha256,
                "metadata_fingerprint": corpus.metadata_fingerprint,
                "match_count": len(corpus.matches),
                "competition_count": len(
                    {match.competition_id for match in corpus.matches}
                ),
                "competition_season_count": len(corpus.selections),
                "first_match_date": corpus.matches[0].match_date.isoformat(),
                "last_match_date": corpus.matches[-1].match_date.isoformat(),
                "event_and_lineup_files_required": corpus.scope.get(
                    "event_and_lineup_files_required"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Training corpus validation failed: {exc}") from exc
