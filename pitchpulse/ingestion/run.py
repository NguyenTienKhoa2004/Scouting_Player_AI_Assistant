"""Ingest all or selected competition-seasons from the pinned dataset."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.shared.paths import PROJECT_ROOT  # noqa: E402
from pitchpulse.ingestion import (  # noqa: E402
    CanonicalEventValidator,
    EventValidationContext,
    IngestionCounts,
    IngestionPlan,
    RawStatsBombValidationError,
    RawStatsBombValidator,
    StatsBombEventNormalizer,
    StatsBombIngestionService,
    StatsBombRawReader,
    build_ingestion_plan,
)
from pitchpulse.dataset.training_dataset_validator import (  # noqa: E402
    load_training_dataset_manifest,
)
from pitchpulse.dataset.bronze import validate_bronze_repository  # noqa: E402
from pitchpulse.ingestion.postgres_writer import (  # noqa: E402
    PostgresDataWriter,
    connect,
)


DEFAULT_DATABASE_URL = "postgresql://pitchpulse:1234567@localhost:5433/pitchpulse"
DEFAULT_DATASET_MANIFEST = (
    PROJECT_ROOT / "configs" / "datasets" / "vaep-training-dataset-v1.json"
)


class MatchIngestionService(Protocol):
    def ingest_match(
        self, ingestion_run_id: int, match_id: int
    ) -> IngestionCounts: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest all matches in the pinned StatsBomb dataset."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
        help=(
            "Versioned dataset to ingest; defaults to the complete pinned VAEP "
            "training dataset."
        ),
    )
    parser.add_argument(
        "--selection",
        action="append",
        metavar="COMPETITION_ID/SEASON_ID",
        help=(
            "Resume only selected competition-seasons; repeat as needed. "
            "Example: --selection 223/282"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest every discovered match, including unchanged matches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ingestion plan without validating or writing match data.",
    )
    return parser.parse_args()


def parse_selection(value: str) -> tuple[int, int]:
    try:
        competition, season = value.split("/", maxsplit=1)
        pair = (int(competition), int(season))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"invalid --selection {value!r}; expected COMPETITION_ID/SEASON_ID"
        ) from exc
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError("--selection IDs must be positive integers")
    return pair


def print_progress(
    position: int,
    total: int,
    match_id: int,
    counts: IngestionCounts,
) -> None:
    print(
        f"[{position:02d}/{total:02d}] match={match_id} "
        f"raw={counts.raw} accepted={counts.accepted} "
        f"rejected={counts.rejected} deduplicated={counts.deduplicated} "
        f"360={counts.raw_360} intervals={counts.raw_lineup_intervals}",
        flush=True,
    )


def ingest_planned_matches(
    connection: object,
    *,
    writer: PostgresDataWriter,
    service: MatchIngestionService,
    plan: IngestionPlan,
    run_id: int,
    source: str,
    dataset_id: str,
    source_version: str,
    competition_id: int,
    season_id: int,
    force: bool = False,
    progress: Callable[[int, int, int, IngestionCounts], None] | None = None,
) -> IngestionCounts:
    """Commit each planned match and its checkpoint in one transaction."""

    counts = IngestionCounts()
    planned_matches = plan.matches_for_ingestion(force=force)
    for position, planned_match in enumerate(planned_matches, start=1):
        if planned_match.content_hash is None:
            raise RuntimeError(
                f"missing content hash for match {planned_match.match_id}"
            )
        with connection.transaction():
            match_counts = service.ingest_match(run_id, planned_match.match_id)
            writer.upsert_ingestion_checkpoint(
                source=source,
                dataset_id=dataset_id,
                competition_id=competition_id,
                season_id=season_id,
                match_id=planned_match.match_id,
                content_hash=planned_match.content_hash,
                source_version=source_version,
                ingestion_run_id=run_id,
            )
        counts.add(match_counts)
        if progress is not None:
            progress(
                position,
                len(planned_matches),
                planned_match.match_id,
                counts,
            )
    return counts


def reconcile_removed_matches(
    connection: object,
    *,
    writer: PostgresDataWriter,
    plan: IngestionPlan,
    run_id: int,
    source: str,
) -> None:
    """Retire match-scoped Silver rows for matches absent from the source."""

    for removed_match in plan.removed:
        with connection.transaction():
            writer.deactivate_match_snapshot(
                ingestion_run_id=run_id,
                source=source,
                match_id=removed_match.match_id,
            )
            writer.delete_ingestion_checkpoint(
                source=source,
                match_id=removed_match.match_id,
            )


def ingest_selection(
    connection: object,
    *,
    reader: StatsBombRawReader,
    dataset_id: str,
    source_version: str,
    manifest_path: str,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    print(f"dataset: {dataset_id}")
    print(
        f"competition-season: {reader.competition_id}/{reader.season_id}"
    )
    writer = PostgresDataWriter(connection)
    checkpoint_hashes = writer.load_ingestion_checkpoint_hashes(
        source="statsbomb",
        competition_id=reader.competition_id,
        season_id=reader.season_id,
    )
    plan = build_ingestion_plan(reader, checkpoint_hashes)
    source_match_ids = plan.source_match_ids
    planned_matches = plan.matches_for_ingestion(force=force)
    match_ids = tuple(match.match_id for match in planned_matches)
    print(f"matches discovered: {len(source_match_ids)}")
    print(
        "ingestion plan: "
        f"new={len(plan.new)} changed={len(plan.changed)} "
        f"unchanged={len(plan.unchanged)} removed={len(plan.removed)}"
    )
    if force:
        print("force mode: all discovered matches will be ingested")
    print(f"matches to ingest: {len(match_ids)}")

    if dry_run:
        return {
            "ingestion_run_id": None,
            "competition_id": reader.competition_id,
            "season_id": reader.season_id,
            "match_count": len(source_match_ids),
            "ingested_match_count": 0,
            "planned_match_count": len(match_ids),
            "new_match_count": len(plan.new),
            "changed_match_count": len(plan.changed),
            "unchanged_match_count": len(plan.unchanged),
            "removed_match_count": len(plan.removed),
            "dry_run": True,
            "raw_validation_passed": None,
            "raw_validation_deferred_issues": 0,
            "raw_events": 0,
            "accepted_events": 0,
            "rejected_events": 0,
            "deduplicated_events": 0,
            "raw_360": 0,
            "accepted_360": 0,
            "raw_lineup_intervals": 0,
            "accepted_lineup_intervals": 0,
            "reconciled": True,
            "database_events": 0,
            "database_360": 0,
            "database_lineup_intervals": 0,
        }

    raw_report = RawStatsBombValidator().validate(reader, match_ids)
    if not raw_report.is_ingestible:
        raise RawStatsBombValidationError(raw_report)
    if raw_report.is_valid:
        print(
            "incremental raw validation passed: "
            f"matches={raw_report.match_count} events={raw_report.event_count} "
            f"lineups={raw_report.lineup_record_count} "
            f"360={raw_report.three_sixty_count}"
        )
    else:
        print(
            "incremental raw validation passed with record-level issues deferred "
            "to quarantine: "
            f"deferred={raw_report.deferred_issue_count} "
            f"reported={len(raw_report.issues)} "
            f"truncated={raw_report.truncated}"
        )

    context = EventValidationContext.from_reader(reader, match_ids)
    run_id = writer.create_ingestion_run(
        dataset_id=dataset_id,
        source="statsbomb",
        source_version=source_version,
        competition_id=reader.competition_id,
        season_id=reader.season_id,
        manifest_path=manifest_path,
        discovered_matches=len(source_match_ids),
        new_matches=len(plan.new),
        changed_matches=len(plan.changed),
        skipped_matches=0 if force else len(plan.unchanged),
        removed_matches=len(plan.removed),
    )
    # Keep a durable audit row even if a long dataset run is interrupted. The
    # match writes themselves remain atomic inside the transaction below.
    connection.commit()
    service = StatsBombIngestionService(
        reader=reader,
        normalizer=StatsBombEventNormalizer(),
        validator=CanonicalEventValidator(context),
        writer=writer,
        project_root=PROJECT_ROOT,
    )

    try:
        reconcile_removed_matches(
            connection,
            writer=writer,
            plan=plan,
            run_id=run_id,
            source="statsbomb",
        )
        counts = ingest_planned_matches(
            connection,
            writer=writer,
            service=service,
            plan=plan,
            run_id=run_id,
            source="statsbomb",
            dataset_id=dataset_id,
            source_version=source_version,
            competition_id=reader.competition_id,
            season_id=reader.season_id,
            force=force,
            progress=print_progress,
        )
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
    except BaseException as exc:
        writer.fail_ingestion_run(run_id, str(exc))
        connection.commit()
        raise

    database_count = writer.count_events_for_matches(source_match_ids)
    database_360_count = writer.count_three_sixty_for_matches(source_match_ids)
    database_interval_count = writer.count_lineup_intervals_for_matches(
        source_match_ids
    )

    return {
        "ingestion_run_id": run_id,
        "competition_id": reader.competition_id,
        "season_id": reader.season_id,
        "match_count": len(source_match_ids),
        "ingested_match_count": len(match_ids),
        "planned_match_count": len(match_ids),
        "new_match_count": len(plan.new),
        "changed_match_count": len(plan.changed),
        "unchanged_match_count": len(plan.unchanged),
        "removed_match_count": len(plan.removed),
        "dry_run": False,
        "raw_validation_passed": raw_report.is_ingestible,
        "raw_validation_deferred_issues": raw_report.deferred_issue_count,
        "raw_events": counts.raw,
        "accepted_events": counts.accepted,
        "rejected_events": counts.rejected,
        "deduplicated_events": counts.deduplicated,
        "raw_360": counts.raw_360,
        "accepted_360": counts.accepted_360,
        "raw_lineup_intervals": counts.raw_lineup_intervals,
        "accepted_lineup_intervals": counts.accepted_lineup_intervals,
        "reconciled": counts.reconciled,
        "database_events": database_count,
        "database_360": database_360_count,
        "database_lineup_intervals": database_interval_count,
    }


def main() -> None:
    args = parse_args()
    dataset = load_training_dataset_manifest(args.dataset_manifest)
    if dataset.bronze is None:
        raise ValueError("ingestion requires a versioned Bronze contract")
    source_commit = dataset.source.get("git_commit")
    if not isinstance(source_commit, str):
        raise ValueError("Bronze source requires a pinned git_commit")
    bronze_report = validate_bronze_repository(
        dataset.bronze.data_root,
        source_commit,
    )
    print(
        "bronze validation passed: "
        f"commit={bronze_report.actual_commit} "
        f"root={bronze_report.repository_root}"
    )
    requested = (
        {parse_selection(value) for value in args.selection}
        if args.selection
        else None
    )
    available = {
        (selection.competition_id, selection.season_id)
        for selection in dataset.selections
    }
    if requested is not None and not requested <= available:
        raise ValueError(
            f"requested selections are not in the dataset: {sorted(requested - available)}"
        )
    source_version = bronze_report.actual_commit
    manifest_path = str(dataset.manifest_path)
    selections = [
        (
            StatsBombRawReader(
                data_root=selection.data_root,
                competition_id=selection.competition_id,
                season_id=selection.season_id,
            ),
            dataset.dataset_id,
            source_version,
            manifest_path,
        )
        for selection in dataset.selections
        if requested is None
        or (selection.competition_id, selection.season_id) in requested
    ]

    results: list[dict[str, object]] = []
    with connect(args.database_url) as connection:
        PostgresDataWriter(connection).require_schema()
        for reader, dataset_id, source_version, manifest_path in selections:
            results.append(
                ingest_selection(
                    connection,
                    reader=reader,
                    dataset_id=dataset_id,
                    source_version=source_version,
                    manifest_path=manifest_path,
                    force=args.force,
                    dry_run=args.dry_run,
                )
            )

    print("\nDry run completed" if args.dry_run else "\nFull ingestion completed")
    print(f"competition_seasons: {len(results)}")
    print(f"matches: {sum(int(item['match_count']) for item in results)}")
    print(f"events: {sum(int(item['accepted_events']) for item in results)}")
    for result in results:
        print(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Full ingestion failed: {exc}") from exc
