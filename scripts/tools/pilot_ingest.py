"""Run the Plan 02 PostgreSQL pilot on two representative matches."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.ingestion import (  # noqa: E402
    CanonicalEventValidator,
    EventValidationContext,
    StatsBombEventNormalizer,
    StatsBombIngestionService,
    StatsBombRawReader,
)
from matchmind.ingestion.postgres_writer import (  # noqa: E402
    PostgresDataWriter,
    connect,
)


DEFAULT_DATABASE_URL = (
    "postgresql://matchmind:1234567@localhost:5433/matchmind"
)
DEFAULT_PILOT_MATCH_IDS = [3857276, 3869685]
DATASET_ID = "statsbomb-open-data-fifa-world-cup-2022"
SOURCE_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
MANIFEST_PATH = "configs/datasets/statsbomb-world-cup-2022.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the two Plan 02 pilot matches.")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument(
        "--match-id",
        type=int,
        action="append",
        dest="match_ids",
        help="Override pilot match IDs; repeat for multiple matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    match_ids = args.match_ids or DEFAULT_PILOT_MATCH_IDS
    reader = StatsBombRawReader()
    context = EventValidationContext.from_reader(reader)

    with connect(args.database_url) as connection:
        writer = PostgresDataWriter(connection)
        writer.require_schema()
        run_id = writer.create_ingestion_run(
            dataset_id=DATASET_ID,
            source="statsbomb",
            source_version=SOURCE_VERSION,
            competition_id=reader.competition_id,
            season_id=reader.season_id,
            manifest_path=MANIFEST_PATH,
        )
        service = StatsBombIngestionService(
            reader=reader,
            normalizer=StatsBombEventNormalizer(),
            validator=CanonicalEventValidator(context),
            writer=writer,
            project_root=PROJECT_ROOT,
        )

        try:
            with connection.transaction():
                counts = service.ingest(run_id, match_ids)
                writer.complete_ingestion_run(
                    run_id,
                    raw_count=counts.raw,
                    accepted_count=counts.accepted,
                    rejected_count=counts.rejected,
                    deduplicated_count=counts.deduplicated,
                    raw_360_count=counts.raw_360,
                    accepted_360_count=counts.accepted_360,
                    rejected_360_count=counts.rejected_360,
                    deduplicated_360_count=counts.deduplicated_360,
                    raw_lineup_interval_count=counts.raw_lineup_intervals,
                    accepted_lineup_interval_count=counts.accepted_lineup_intervals,
                    rejected_lineup_interval_count=counts.rejected_lineup_intervals,
                    deduplicated_lineup_interval_count=(
                        counts.deduplicated_lineup_intervals
                    ),
                )
        except Exception as exc:
            writer.fail_ingestion_run(run_id, str(exc))
            raise

        database_count = writer.count_events_for_matches(match_ids)
        database_360_count = writer.count_three_sixty_for_matches(match_ids)
        database_interval_count = writer.count_lineup_intervals_for_matches(match_ids)

    print(f"ingestion_run_id: {run_id}")
    print(f"match_ids: {match_ids}")
    print(f"raw_events: {counts.raw}")
    print(f"accepted_events: {counts.accepted}")
    print(f"rejected_events: {counts.rejected}")
    print(f"deduplicated_events: {counts.deduplicated}")
    print(f"raw_360: {counts.raw_360}")
    print(f"accepted_360: {counts.accepted_360}")
    print(f"rejected_360: {counts.rejected_360}")
    print(f"deduplicated_360: {counts.deduplicated_360}")
    print(f"raw_lineup_intervals: {counts.raw_lineup_intervals}")
    print(f"reconciled: {counts.reconciled}")
    print(f"database_events_for_pilot: {database_count}")
    print(f"database_360_for_pilot: {database_360_count}")
    print(f"database_lineup_intervals_for_pilot: {database_interval_count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Pilot ingestion failed: {exc}") from exc
