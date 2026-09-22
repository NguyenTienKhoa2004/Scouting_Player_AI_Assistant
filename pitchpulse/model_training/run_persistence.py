"""Persist the completed VAEP run to PostgreSQL."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.model_training.postgres_writer import PostgresVaepWriter  # noqa: E402
from pitchpulse.pipelines.settings import MODEL_OUTPUT, latest_artifact  # noqa: E402


DEFAULT_DATABASE_URL = "postgresql://pitchpulse:1234567@localhost:5433/pitchpulse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persist completed VAEP artifacts to PostgreSQL"
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        help="Plan 04 artifact directory; defaults to the latest completed run",
    )
    parser.add_argument(
        "--analytics-run-id",
        type=int,
        default=(
            int(os.environ["PITCHPULSE_ANALYTICS_RUN_ID"])
            if os.environ.get("PITCHPULSE_ANALYTICS_RUN_ID")
            else None
        ),
        help=(
            "Override the real Plan 03 analytics run for this dataset. Normally "
            "discovered from the registered Plan 03 manifest."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_directory = args.artifact_directory or latest_artifact(
        MODEL_OUTPUT, "training_manifest.json"
    )
    with psycopg.connect(args.database_url, autocommit=True) as connection:
        result = PostgresVaepWriter(connection).persist(
            artifact_directory,
            analytics_run_id=args.analytics_run_id,
            batch_size=args.batch_size,
            progress=lambda message: print(message, flush=True),
        )
    outcome = "created" if result.created else "already existed"
    print(
        f"VAEP modeling run {result.modeling_run_id} {outcome}; "
        f"analytics run {result.analytics_run_id}; manifest: {result.manifest}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"VAEP persistence failed: {exc}") from exc
