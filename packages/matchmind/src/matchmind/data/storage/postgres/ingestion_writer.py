"""PostgreSQL persistence for normalized and validated StatsBomb data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg
from psycopg import Connection
from psycopg.types.json import Jsonb

from matchmind.data.ingestion.normalizer import (
    Canonical360Frame,
    CanonicalEvent,
    CanonicalLineupInterval,
)
from matchmind.data.ingestion.reader import RawRecord


@dataclass(frozen=True, slots=True)
class EventUpsertCounts:
    inserted: int
    updated: int


class PostgresDataWriter:
    """Write dimensions, events, invalid records, and ingestion metadata."""

    EVENT_UPSERT_SQL = """
        INSERT INTO events (
            source, source_event_id, ingestion_run_id, match_id, team_id,
            player_id, event_type, period, timestamp, minute, second,
            x, y, end_x, end_y, outcome, xg,
            source_record_index, source_event_index,
            possession_id, possession_team_id, event_subtype, duration,
            body_part, recipient_id, play_pattern, under_pressure,
            counterpress, related_event_ids, raw_details
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (source, source_event_id) DO UPDATE SET
            ingestion_run_id = EXCLUDED.ingestion_run_id,
            match_id = EXCLUDED.match_id,
            team_id = EXCLUDED.team_id,
            player_id = EXCLUDED.player_id,
            event_type = EXCLUDED.event_type,
            period = EXCLUDED.period,
            timestamp = EXCLUDED.timestamp,
            minute = EXCLUDED.minute,
            second = EXCLUDED.second,
            x = EXCLUDED.x,
            y = EXCLUDED.y,
            end_x = EXCLUDED.end_x,
            end_y = EXCLUDED.end_y,
            outcome = EXCLUDED.outcome,
            xg = EXCLUDED.xg,
            source_record_index = EXCLUDED.source_record_index,
            source_event_index = EXCLUDED.source_event_index,
            possession_id = EXCLUDED.possession_id,
            possession_team_id = EXCLUDED.possession_team_id,
            event_subtype = EXCLUDED.event_subtype,
            duration = EXCLUDED.duration,
            body_part = EXCLUDED.body_part,
            recipient_id = EXCLUDED.recipient_id,
            play_pattern = EXCLUDED.play_pattern,
            under_pressure = EXCLUDED.under_pressure,
            counterpress = EXCLUDED.counterpress,
            related_event_ids = EXCLUDED.related_event_ids,
            raw_details = EXCLUDED.raw_details
    """

    def __init__(self, connection: Connection[Any]) -> None:
        self.connection = connection

    def require_schema(self) -> None:
        row = self.connection.execute(
            "SELECT to_regclass('public.events'), "
            "to_regclass('public.ingestion_runs'), "
            "to_regclass('public.player_match_intervals'), "
            "to_regclass('public.event_360')"
        ).fetchone()
        if row is None or any(value is None for value in row):
            raise RuntimeError(
                "VAEP-ready Plan 02 schema is missing. Apply migrations 001 and 002 first."
            )

    def create_ingestion_run(
        self,
        *,
        dataset_id: str,
        source: str,
        source_version: str,
        competition_id: int,
        season_id: int,
        manifest_path: str,
    ) -> int:
        row = self.connection.execute(
            """
            INSERT INTO ingestion_runs (
                dataset_id, source, source_version, competition_id,
                season_id, status, manifest_path
            ) VALUES (%s, %s, %s, %s, %s, 'running', %s)
            RETURNING id
            """,
            (
                dataset_id,
                source,
                source_version,
                competition_id,
                season_id,
                manifest_path,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL did not return an ingestion run ID")
        return int(row[0])

    def complete_ingestion_run(
        self,
        run_id: int,
        *,
        raw_count: int,
        accepted_count: int,
        rejected_count: int,
        deduplicated_count: int,
        raw_360_count: int = 0,
        accepted_360_count: int = 0,
        rejected_360_count: int = 0,
        deduplicated_360_count: int = 0,
        raw_lineup_interval_count: int = 0,
        accepted_lineup_interval_count: int = 0,
        rejected_lineup_interval_count: int = 0,
        deduplicated_lineup_interval_count: int = 0,
    ) -> None:
        self.connection.execute(
            """
            UPDATE ingestion_runs
            SET status = 'succeeded',
                raw_count = %s,
                accepted_count = %s,
                rejected_count = %s,
                deduplicated_count = %s,
                raw_360_count = %s,
                accepted_360_count = %s,
                rejected_360_count = %s,
                deduplicated_360_count = %s,
                raw_lineup_interval_count = %s,
                accepted_lineup_interval_count = %s,
                rejected_lineup_interval_count = %s,
                deduplicated_lineup_interval_count = %s,
                completed_at = CURRENT_TIMESTAMP,
                error_message = NULL
            WHERE id = %s
            """,
            (
                raw_count,
                accepted_count,
                rejected_count,
                deduplicated_count,
                raw_360_count,
                accepted_360_count,
                rejected_360_count,
                deduplicated_360_count,
                raw_lineup_interval_count,
                accepted_lineup_interval_count,
                rejected_lineup_interval_count,
                deduplicated_lineup_interval_count,
                run_id,
            ),
        )

    def fail_ingestion_run(self, run_id: int, error_message: str) -> None:
        self.connection.execute(
            """
            UPDATE ingestion_runs
            SET status = 'failed',
                completed_at = CURRENT_TIMESTAMP,
                error_message = %s
            WHERE id = %s
            """,
            (error_message[:4000], run_id),
        )

    def upsert_teams(self, teams: Iterable[tuple[int, str]]) -> None:
        unique = {team_id: name for team_id, name in teams}
        if not unique:
            return
        self.connection.cursor().executemany(
            """
            INSERT INTO teams (id, name) VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
            """,
            sorted(unique.items()),
        )

    def upsert_players(
        self, players: Iterable[tuple[int, str, int, str | None]]
    ) -> None:
        unique = {player_id: (name, team_id, position) for player_id, name, team_id, position in players}
        if not unique:
            return
        values = [
            (player_id, name, team_id, position)
            for player_id, (name, team_id, position) in sorted(unique.items())
        ]
        self.connection.cursor().executemany(
            """
            INSERT INTO players (id, name, team_id, position)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                team_id = EXCLUDED.team_id,
                position = COALESCE(EXCLUDED.position, players.position)
            """,
            values,
        )

    def upsert_match(
        self,
        *,
        match_id: int,
        competition_id: int,
        competition: str,
        season_id: int,
        season: str,
        home_team_id: int,
        away_team_id: int,
        home_score: int,
        away_score: int,
        match_date: Any,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO matches (
                id, competition_id, competition, season_id, season,
                home_team_id, away_team_id, home_score, away_score, match_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                competition_id = EXCLUDED.competition_id,
                competition = EXCLUDED.competition,
                season_id = EXCLUDED.season_id,
                season = EXCLUDED.season,
                home_team_id = EXCLUDED.home_team_id,
                away_team_id = EXCLUDED.away_team_id,
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                match_date = EXCLUDED.match_date
            """,
            (
                match_id,
                competition_id,
                competition,
                season_id,
                season,
                home_team_id,
                away_team_id,
                home_score,
                away_score,
                match_date,
            ),
        )

    def upsert_events(
        self, ingestion_run_id: int, events: Sequence[CanonicalEvent]
    ) -> EventUpsertCounts:
        if not events:
            return EventUpsertCounts(inserted=0, updated=0)

        unique_events: dict[tuple[str, UUID], CanonicalEvent] = {}
        for event in events:
            unique_events[(event.source, event.source_event_id)] = event

        by_source: dict[str, list[UUID]] = {}
        for source, event_id in unique_events:
            by_source.setdefault(source, []).append(event_id)

        existing: set[tuple[str, UUID]] = set()
        for source, event_ids in by_source.items():
            rows = self.connection.execute(
                """
                SELECT source, source_event_id
                FROM events
                WHERE source = %s AND source_event_id = ANY(%s::uuid[])
                """,
                (source, event_ids),
            ).fetchall()
            existing.update((str(row[0]), row[1]) for row in rows)

        values = [
            (
                event.source,
                event.source_event_id,
                ingestion_run_id,
                event.match_id,
                event.team_id,
                event.player_id,
                event.event_type,
                event.period,
                event.timestamp,
                event.minute,
                event.second,
                event.x,
                event.y,
                event.end_x,
                event.end_y,
                event.outcome,
                event.xg,
                event.source_record_index,
                event.source_event_index,
                event.possession_id,
                event.possession_team_id,
                event.event_subtype,
                event.duration,
                event.body_part,
                event.recipient_id,
                event.play_pattern,
                event.under_pressure,
                event.counterpress,
                list(event.related_event_ids),
                Jsonb(event.raw_details),
            )
            for event in unique_events.values()
        ]
        self.connection.cursor().executemany(self.EVENT_UPSERT_SQL, values)

        updated = sum(key in existing for key in unique_events)
        return EventUpsertCounts(
            inserted=len(unique_events) - updated,
            updated=updated,
        )

    def upsert_player_match_intervals(
        self,
        ingestion_run_id: int,
        intervals: Sequence[CanonicalLineupInterval],
    ) -> EventUpsertCounts:
        if not intervals:
            return EventUpsertCounts(inserted=0, updated=0)
        unique = {
            (
                item.source,
                item.match_id,
                item.player_id,
                item.position_id,
                item.from_seconds,
            ): item
            for item in intervals
        }
        match_ids = sorted({item.match_id for item in unique.values()})
        rows = self.connection.execute(
            """
            SELECT source, match_id, player_id, position_id, from_seconds
            FROM player_match_intervals
            WHERE match_id = ANY(%s::bigint[])
            """,
            (match_ids,),
        ).fetchall()
        existing = {
            (str(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))
            for row in rows
        }
        values = [
            (
                ingestion_run_id,
                item.source,
                item.source_record_index,
                item.match_id,
                item.team_id,
                item.player_id,
                item.position_id,
                item.position,
                item.from_seconds,
                item.to_seconds,
                item.from_period,
                item.to_period,
                item.start_reason,
                item.end_reason,
                Jsonb(item.raw_details),
            )
            for item in unique.values()
        ]
        self.connection.cursor().executemany(
            """
            INSERT INTO player_match_intervals (
                ingestion_run_id, source, source_record_index,
                match_id, team_id, player_id, position_id, position,
                from_seconds, to_seconds, from_period, to_period,
                start_reason, end_reason, raw_details
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (source, match_id, player_id, position_id, from_seconds)
            DO UPDATE SET
                ingestion_run_id = EXCLUDED.ingestion_run_id,
                source_record_index = EXCLUDED.source_record_index,
                team_id = EXCLUDED.team_id,
                position = EXCLUDED.position,
                to_seconds = EXCLUDED.to_seconds,
                from_period = EXCLUDED.from_period,
                to_period = EXCLUDED.to_period,
                start_reason = EXCLUDED.start_reason,
                end_reason = EXCLUDED.end_reason,
                raw_details = EXCLUDED.raw_details
            """,
            values,
        )
        updated = sum(key in existing for key in unique)
        return EventUpsertCounts(inserted=len(unique) - updated, updated=updated)

    def upsert_three_sixty(
        self,
        ingestion_run_id: int,
        frames: Sequence[Canonical360Frame],
    ) -> EventUpsertCounts:
        if not frames:
            return EventUpsertCounts(inserted=0, updated=0)
        unique = {
            (frame.source, frame.source_event_id): frame for frame in frames
        }
        by_source: dict[str, list[UUID]] = {}
        for source, event_id in unique:
            by_source.setdefault(source, []).append(event_id)
        existing: set[tuple[str, UUID]] = set()
        for source, event_ids in by_source.items():
            rows = self.connection.execute(
                """
                SELECT source, source_event_id
                FROM event_360
                WHERE source = %s AND source_event_id = ANY(%s::uuid[])
                """,
                (source, event_ids),
            ).fetchall()
            existing.update((str(row[0]), row[1]) for row in rows)
        values = [
            (
                frame.source,
                frame.source_event_id,
                ingestion_run_id,
                frame.source_record_index,
                frame.match_id,
                Jsonb(list(frame.visible_area)),
                Jsonb(list(frame.freeze_frame)),
                Jsonb(frame.raw_details),
            )
            for frame in unique.values()
        ]
        self.connection.cursor().executemany(
            """
            INSERT INTO event_360 (
                source, source_event_id, ingestion_run_id,
                source_record_index, match_id,
                visible_area, freeze_frame, raw_details
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, source_event_id) DO UPDATE SET
                ingestion_run_id = EXCLUDED.ingestion_run_id,
                source_record_index = EXCLUDED.source_record_index,
                match_id = EXCLUDED.match_id,
                visible_area = EXCLUDED.visible_area,
                freeze_frame = EXCLUDED.freeze_frame,
                raw_details = EXCLUDED.raw_details
            """,
            values,
        )
        updated = sum(key in existing for key in unique)
        return EventUpsertCounts(inserted=len(unique) - updated, updated=updated)

    def insert_invalid_event(
        self,
        *,
        ingestion_run_id: int,
        record: RawRecord,
        reason_code: str,
        reason: str,
        project_root: Path,
    ) -> None:
        try:
            source_file = str(record.source_file.resolve().relative_to(project_root.resolve()))
        except ValueError:
            source_file = str(record.source_file)
        self.connection.execute(
            """
            INSERT INTO invalid_events (
                ingestion_run_id, source, source_file, source_record_index,
                source_record, reason_code, reason
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                ingestion_run_id,
                record.source,
                source_file,
                record.source_record_index,
                Jsonb(record.payload),
                reason_code,
                reason,
            ),
        )

    def count_events_for_matches(self, match_ids: Sequence[int]) -> int:
        if not match_ids:
            return 0
        row = self.connection.execute(
            "SELECT COUNT(*) FROM events WHERE match_id = ANY(%s::bigint[])",
            (list(match_ids),),
        ).fetchone()
        return int(row[0]) if row else 0

    def count_three_sixty_for_matches(self, match_ids: Sequence[int]) -> int:
        if not match_ids:
            return 0
        row = self.connection.execute(
            "SELECT COUNT(*) FROM event_360 WHERE match_id = ANY(%s::bigint[])",
            (list(match_ids),),
        ).fetchone()
        return int(row[0]) if row else 0

    def count_lineup_intervals_for_matches(self, match_ids: Sequence[int]) -> int:
        if not match_ids:
            return 0
        row = self.connection.execute(
            "SELECT COUNT(*) FROM player_match_intervals "
            "WHERE match_id = ANY(%s::bigint[])",
            (list(match_ids),),
        ).fetchone()
        return int(row[0]) if row else 0


def connect(database_url: str) -> Connection[Any]:
    """Create an autocommit connection used by the ingestion transaction."""
    return psycopg.connect(database_url, autocommit=True)
