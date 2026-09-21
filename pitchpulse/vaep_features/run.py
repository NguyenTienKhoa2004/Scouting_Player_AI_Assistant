"""Build a versioned SPADL and VAEP feature corpus from PostgreSQL events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.pipelines.settings import (  # noqa: E402
    CORPUS_MANIFEST,
    FEATURE_OUTPUT,
)
from pitchpulse.vaep_features import AnalyticsDatasetBuilder  # noqa: E402
from pitchpulse.corpus.training_dataset_validator import (  # noqa: E402
    load_training_corpus_manifest,
)
from pitchpulse.vaep_features.artifacts import ParquetArtifactWriter  # noqa: E402
from pitchpulse.spadl.input_reader import (  # noqa: E402
    PostgresSpadlInputReader,
)
from pitchpulse.ingestion.postgres_writer import connect  # noqa: E402


DEFAULT_DATABASE_URL = "postgresql://pitchpulse:1234567@localhost:5433/pitchpulse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build versioned SPADL actions and leakage-safe features."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=CORPUS_MANIFEST,
        help="Versioned training corpus manifest to build.",
    )
    parser.add_argument(
        "--include-360",
        action="store_true",
        help="Use the separate 360 feature contract.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FEATURE_OUTPUT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help=(
            "Matches held in memory per batch for corpus builds (default: 5, "
            "safe for an 8 GB machine)."
        ),
    )
    return parser.parse_args()


def build_corpus(
    args: argparse.Namespace, match_ids: list[int]
) -> tuple[object, dict[str, object]]:
    total_batches = (len(match_ids) + args.batch_size - 1) // args.batch_size
    with connect(args.database_url) as connection:
        reader = PostgresSpadlInputReader(connection)
        builder = AnalyticsDatasetBuilder()

        def datasets():
            for offset in range(0, len(match_ids), args.batch_size):
                batch_number = offset // args.batch_size + 1
                batch_ids = match_ids[offset : offset + args.batch_size]
                print(
                    f"building batch {batch_number}/{total_batches}: "
                    f"{len(batch_ids)} matches",
                    flush=True,
                )
                inputs = reader.load(batch_ids)
                yield builder.build(inputs, include_360=args.include_360)

        artifacts = ParquetArtifactWriter().write_many(
            datasets(),
            args.output_dir,
            progress=lambda message: print(message, flush=True),
        )

    quality_report = json.loads(
        artifacts.quality_report.read_text(encoding="utf-8")
    )
    return artifacts, quality_report


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    corpus = load_training_corpus_manifest(args.corpus_manifest)
    match_ids = sorted(corpus.match_ids)
    artifacts, quality_report = build_corpus(args, match_ids)

    print(
        json.dumps(
            {
                "analytics_run_id": None,
                "artifact_directory": str(artifacts.directory),
                **quality_report,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Analytics build failed: {exc}") from exc
