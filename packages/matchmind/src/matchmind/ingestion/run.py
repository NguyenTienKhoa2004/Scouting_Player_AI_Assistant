"""Ingest all or selected competition-seasons from the pinned corpus."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PACKAGE_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_SRC))

from matchmind.shared.paths import PROJECT_ROOT  # noqa: E402
from matchmind.ingestion import (  # noqa: E402
    CanonicalEventValidator,
    EventValidationContext,
    IngestionCounts,
    StatsBombEventNormalizer,
    StatsBombIngestionService,
    StatsBombRawReader,
)
from matchmind.corpus.training_dataset_validator import (  # noqa: E402
    load_training_corpus_manifest,
)
from matchmind.ingestion.postgres_writer import (  # noqa: E402
    PostgresDataWriter,
    connect,
)


DEFAULT_DATABASE_URL = "postgresql://matchmind:1234567@localhost:5433/matchmind"
DEFAULT_CORPUS_MANIFEST = (
    PROJECT_ROOT / "configs" / "datasets" / "vaep-training-corpus-v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest all matches in the pinned StatsBomb dataset."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_MANIFEST,
        help=(
            "Versioned corpus to ingest; defaults to the complete pinned VAEP "
            "training corpus."
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


def ingest_selection(
    connection: object,
    *,
    reader: StatsBombRawReader,
    dataset_id: str,
    source_version: str,
    manifest_path: str,
) -> dict[str, object]:
    match_ids = [record.match_id for record in reader.iter_matches()]
    context = EventValidationContext.from_reader(reader)

    print(f"dataset: {dataset_id}")
    print(
        f"competition-season: {reader.competition_id}/{reader.season_id}"
    )
    print(f"matches selected: {len(match_ids)}")

    writer = PostgresDataWriter(connection)
    run_id = writer.create_ingestion_run(
        dataset_id=dataset_id,
        source="statsbomb",
        source_version=source_version,
        competition_id=reader.competition_id,
        season_id=reader.season_id,
        manifest_path=manifest_path,
    )
    # Keep a durable audit row even if a long corpus run is interrupted. The
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
        with connection.transaction():
            counts = service.ingest(
                run_id,
                match_ids,
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

    database_count = writer.count_events_for_matches(match_ids)
    database_360_count = writer.count_three_sixty_for_matches(match_ids)
    database_interval_count = writer.count_lineup_intervals_for_matches(match_ids)

    return {
        "ingestion_run_id": run_id,
        "competition_id": reader.competition_id,
        "season_id": reader.season_id,
        "match_count": len(match_ids),
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
    corpus = load_training_corpus_manifest(args.corpus_manifest)
    requested = (
        {parse_selection(value) for value in args.selection}
        if args.selection
        else None
    )
    available = {
        (selection.competition_id, selection.season_id)
        for selection in corpus.selections
    }
    if requested is not None and not requested <= available:
        raise ValueError(
            f"requested selections are not in the corpus: {sorted(requested - available)}"
        )
    source_version = str(corpus.source.get("git_commit") or "unknown")
    manifest_path = str(corpus.manifest_path)
    selections = [
        (
            StatsBombRawReader(
                data_root=selection.matches_path.parents[2],
                competition_id=selection.competition_id,
                season_id=selection.season_id,
            ),
            corpus.corpus_id,
            source_version,
            manifest_path,
        )
        for selection in corpus.selections
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
                )
            )

    print("\nFull ingestion completed")
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
