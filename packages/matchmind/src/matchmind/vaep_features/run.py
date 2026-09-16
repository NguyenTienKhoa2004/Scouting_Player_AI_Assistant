"""Build SPADL actions and VAEP features from PostgreSQL events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PACKAGE_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_SRC))

from matchmind.shared.paths import PROJECT_ROOT  # noqa: E402
from matchmind.vaep_features import AnalyticsDatasetBuilder  # noqa: E402
from matchmind.corpus.training_dataset_validator import (  # noqa: E402
    load_training_corpus_manifest,
)
from matchmind.vaep_features.artifacts import (  # noqa: E402
    ChunkedParquetArtifactWriter,
    ParquetArtifactWriter,
)
from matchmind.vaep_features.postgres_writer import (  # noqa: E402
    PostgresAnalyticsWriter,
)
from matchmind.spadl.input_reader import (  # noqa: E402
    PostgresSpadlInputReader,
)
from matchmind.ingestion.postgres_writer import connect  # noqa: E402


DEFAULT_DATABASE_URL = "postgresql://matchmind:1234567@localhost:5433/matchmind"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build versioned SPADL actions and leakage-safe features."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument(
        "--match-id",
        type=int,
        action="append",
        dest="match_ids",
        help="Build selected matches; repeat for multiple matches.",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        help=(
            "Build exactly the matches declared by a versioned training corpus; "
            "cannot be combined with --match-id."
        ),
    )
    parser.add_argument(
        "--include-360",
        action="store_true",
        help="Use the separate 360 feature contract.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "features" / "plan03",
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

        artifacts = ChunkedParquetArtifactWriter().write(
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
    if args.match_ids and args.corpus_manifest:
        raise ValueError("--match-id and --corpus-manifest are mutually exclusive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    match_ids = args.match_ids
    if args.corpus_manifest:
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
        return

    with connect(args.database_url) as connection:
        inputs = PostgresSpadlInputReader(connection).load(match_ids)
        dataset = AnalyticsDatasetBuilder().build(
            inputs, include_360=args.include_360
        )
        artifacts = ParquetArtifactWriter().write(dataset, args.output_dir)

        writer = PostgresAnalyticsWriter(connection)
        writer.require_schema()
        selected_matches = sorted({event.match_id for event in inputs.events})
        with connection.transaction():
            run_id = writer.create_run(
                dataset, selected_matches, artifacts.directory
            )
            writer.write_dataset(run_id, dataset)
            writer.complete_run(run_id, dataset)

    print(
        json.dumps(
            {
                "analytics_run_id": run_id,
                "artifact_directory": str(artifacts.directory),
                **dataset.quality_report(),
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
