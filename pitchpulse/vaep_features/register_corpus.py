"""Register the latest Parquet-first Plan 03 corpus in PostgreSQL."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.pipelines.settings import FEATURE_OUTPUT, latest_artifact  # noqa: E402
from pitchpulse.vaep_features.artifact_registration import (  # noqa: E402
    PostgresAnalyticsArtifactRegistrar,
)


DEFAULT_DATABASE_URL = "postgresql://pitchpulse:1234567@localhost:5433/pitchpulse"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a Parquet-first Plan 03 corpus in PostgreSQL"
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    artifact_directory = args.artifact_directory or latest_artifact(
        FEATURE_OUTPUT, "manifest.json"
    )
    with psycopg.connect(args.database_url, autocommit=True) as connection:
        result = PostgresAnalyticsArtifactRegistrar(connection).register(
            artifact_directory,
            batch_size=args.batch_size,
            progress=lambda message: print(message, flush=True),
        )
    outcome = "created" if result.created else "already existed"
    print(f"Analytics run {result.analytics_run_id} {outcome}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Plan 03 registration failed: {exc}") from exc
