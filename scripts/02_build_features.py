"""Build SPADL actions and VAEP features from PostgreSQL events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matchmind.feature_engineering import AnalyticsPipeline  # noqa: E402
from matchmind.storage.parquet import ParquetArtifactWriter  # noqa: E402
from matchmind.storage.postgres.analytics_writer import (  # noqa: E402
    PostgresAnalyticsWriter,
)
from matchmind.storage.postgres.feature_reader import (  # noqa: E402
    PostgresSpadlInputReader,
)
from matchmind.storage.postgres.ingestion_writer import connect  # noqa: E402


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
        "--include-360",
        action="store_true",
        help="Use the separate 360 feature contract.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "plan03",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with connect(args.database_url) as connection:
        inputs = PostgresSpadlInputReader(connection).load(args.match_ids)
        dataset = AnalyticsPipeline().build(inputs, include_360=args.include_360)
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
