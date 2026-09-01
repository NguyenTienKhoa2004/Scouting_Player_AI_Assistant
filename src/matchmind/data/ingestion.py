"""Orchestrate StatsBomb raw reading, normalization, validation, and upsert."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from .normalizer import (
    NormalizationError,
    StatsBomb360Normalizer,
    StatsBombEventNormalizer,
    StatsBombLineupNormalizer,
)
from .postgres_writer import PostgresDataWriter
from .raw_reader import RawDataFormatError, RawMatchBundle, StatsBombRawReader
from .validator import (
    Canonical360Validator,
    CanonicalEventValidator,
    CanonicalLineupIntervalValidator,
)


@dataclass(slots=True)
class IngestionCounts:
    raw: int = 0
    accepted: int = 0
    rejected: int = 0
    deduplicated: int = 0
    raw_360: int = 0
    accepted_360: int = 0
    rejected_360: int = 0
    deduplicated_360: int = 0
    raw_lineup_intervals: int = 0
    accepted_lineup_intervals: int = 0
    rejected_lineup_intervals: int = 0
    deduplicated_lineup_intervals: int = 0

    @property
    def reconciled(self) -> bool:
        events_ok = self.raw == self.accepted + self.rejected + self.deduplicated
        frames_ok = self.raw_360 == (
            self.accepted_360 + self.rejected_360 + self.deduplicated_360
        )
        intervals_ok = self.raw_lineup_intervals == (
            self.accepted_lineup_intervals
            + self.rejected_lineup_intervals
            + self.deduplicated_lineup_intervals
        )
        return events_ok and frames_ok and intervals_ok


class StatsBombIngestionService:
    """Ingest selected matches atomically through the canonical pipeline."""

    def __init__(
        self,
        *,
        reader: StatsBombRawReader,
        normalizer: StatsBombEventNormalizer,
        validator: CanonicalEventValidator,
        writer: PostgresDataWriter,
        project_root: Path,
        lineup_normalizer: StatsBombLineupNormalizer | None = None,
        lineup_validator: CanonicalLineupIntervalValidator | None = None,
        three_sixty_normalizer: StatsBomb360Normalizer | None = None,
        three_sixty_validator: Canonical360Validator | None = None,
    ) -> None:
        self.reader = reader
        self.normalizer = normalizer
        self.validator = validator
        self.writer = writer
        self.project_root = project_root
        self.lineup_normalizer = lineup_normalizer or StatsBombLineupNormalizer()
        self.lineup_validator = lineup_validator or CanonicalLineupIntervalValidator()
        self.three_sixty_normalizer = (
            three_sixty_normalizer or StatsBomb360Normalizer()
        )
        self.three_sixty_validator = three_sixty_validator or Canonical360Validator()

    def ingest(
        self,
        ingestion_run_id: int,
        match_ids: list[int],
        progress: Callable[[int, int, int, IngestionCounts], None] | None = None,
    ) -> IngestionCounts:
        counts = IngestionCounts()
        seen_source_events: set[tuple[str, Any]] = set()
        seen_three_sixty: set[tuple[str, Any]] = set()

        for position, match_id in enumerate(match_ids, start=1):
            bundle = self.reader.read_match_bundle(match_id)
            self._upsert_dimensions(bundle)
            self._ingest_lineup_intervals(ingestion_run_id, bundle, counts)
            valid_events = []

            for record in bundle.events:
                counts.raw += 1
                try:
                    event = self.normalizer.normalize(record)
                except NormalizationError as exc:
                    counts.rejected += 1
                    self.writer.insert_invalid_event(
                        ingestion_run_id=ingestion_run_id,
                        record=record,
                        reason_code=exc.code,
                        reason=exc.message,
                        project_root=self.project_root,
                    )
                    continue

                source_key = (event.source, event.source_event_id)
                if source_key in seen_source_events:
                    counts.deduplicated += 1
                    continue
                seen_source_events.add(source_key)

                result = self.validator.validate(event)
                if not result.is_valid:
                    counts.rejected += 1
                    reason_code = (
                        result.issues[0].code
                        if len(result.issues) == 1
                        else "multiple_validation_errors"
                    )
                    reason = "; ".join(
                        f"{issue.code}: {issue.message}" for issue in result.issues
                    )
                    self.writer.insert_invalid_event(
                        ingestion_run_id=ingestion_run_id,
                        record=record,
                        reason_code=reason_code,
                        reason=reason,
                        project_root=self.project_root,
                    )
                    continue

                valid_events.append(event)

            self._validate_event_order(valid_events, match_id)
            upserted = self.writer.upsert_events(ingestion_run_id, valid_events)
            counts.accepted += upserted.inserted
            counts.deduplicated += upserted.updated
            self._ingest_three_sixty(
                ingestion_run_id,
                bundle,
                {event.source_event_id for event in valid_events},
                seen_three_sixty,
                counts,
            )
            if progress is not None:
                progress(position, len(match_ids), match_id, counts)

        if not counts.reconciled:
            raise RuntimeError(
                "Ingestion counts do not reconcile: "
                f"raw={counts.raw}, accepted={counts.accepted}, "
                f"rejected={counts.rejected}, deduplicated={counts.deduplicated}; "
                f"raw_360={counts.raw_360}, accepted_360={counts.accepted_360}, "
                f"rejected_360={counts.rejected_360}, "
                f"deduplicated_360={counts.deduplicated_360}; "
                f"raw_lineup_intervals={counts.raw_lineup_intervals}, "
                f"accepted_lineup_intervals={counts.accepted_lineup_intervals}, "
                f"rejected_lineup_intervals={counts.rejected_lineup_intervals}, "
                f"deduplicated_lineup_intervals="
                f"{counts.deduplicated_lineup_intervals}"
            )
        return counts

    def _ingest_lineup_intervals(
        self,
        ingestion_run_id: int,
        bundle: RawMatchBundle,
        counts: IngestionCounts,
    ) -> None:
        valid_intervals = []
        for record in bundle.lineups:
            try:
                raw_interval_count = self._lineup_interval_count(record.payload)
            except RawDataFormatError as exc:
                counts.raw_lineup_intervals += 1
                counts.rejected_lineup_intervals += 1
                self.writer.insert_invalid_event(
                    ingestion_run_id=ingestion_run_id,
                    record=record,
                    reason_code="malformed_lineup",
                    reason=str(exc),
                    project_root=self.project_root,
                )
                continue
            counts.raw_lineup_intervals += raw_interval_count
            try:
                intervals = self.lineup_normalizer.normalize(record)
            except NormalizationError as exc:
                counts.rejected_lineup_intervals += raw_interval_count
                self.writer.insert_invalid_event(
                    ingestion_run_id=ingestion_run_id,
                    record=record,
                    reason_code=exc.code,
                    reason=exc.message,
                    project_root=self.project_root,
                )
                continue
            for interval in intervals:
                result = self.lineup_validator.validate(interval)
                if result.is_valid:
                    valid_intervals.append(interval)
                    continue
                counts.rejected_lineup_intervals += 1
                reason = "; ".join(
                    f"{issue.code}: {issue.message}" for issue in result.issues
                )
                self.writer.insert_invalid_event(
                    ingestion_run_id=ingestion_run_id,
                    record=record,
                    reason_code=result.issues[0].code,
                    reason=reason,
                    project_root=self.project_root,
                )
        upserted = self.writer.upsert_player_match_intervals(
            ingestion_run_id, valid_intervals
        )
        counts.accepted_lineup_intervals += upserted.inserted
        counts.deduplicated_lineup_intervals += upserted.updated

    def _ingest_three_sixty(
        self,
        ingestion_run_id: int,
        bundle: RawMatchBundle,
        known_event_ids: set[Any],
        seen: set[tuple[str, Any]],
        counts: IngestionCounts,
    ) -> None:
        valid_frames = []
        for record in bundle.three_sixty:
            counts.raw_360 += 1
            try:
                frame = self.three_sixty_normalizer.normalize(record)
            except NormalizationError as exc:
                counts.rejected_360 += 1
                self.writer.insert_invalid_event(
                    ingestion_run_id=ingestion_run_id,
                    record=record,
                    reason_code=exc.code,
                    reason=exc.message,
                    project_root=self.project_root,
                )
                continue
            source_key = (frame.source, frame.source_event_id)
            if source_key in seen:
                counts.deduplicated_360 += 1
                continue
            seen.add(source_key)
            result = self.three_sixty_validator.validate(
                frame, known_event_ids=known_event_ids
            )
            if not result.is_valid:
                counts.rejected_360 += 1
                reason = "; ".join(
                    f"{issue.code}: {issue.message}" for issue in result.issues
                )
                self.writer.insert_invalid_event(
                    ingestion_run_id=ingestion_run_id,
                    record=record,
                    reason_code=(
                        result.issues[0].code
                        if len(result.issues) == 1
                        else "multiple_validation_errors"
                    ),
                    reason=reason,
                    project_root=self.project_root,
                )
                continue
            valid_frames.append(frame)
        upserted = self.writer.upsert_three_sixty(
            ingestion_run_id, valid_frames
        )
        counts.accepted_360 += upserted.inserted
        counts.deduplicated_360 += upserted.updated

    @staticmethod
    def _validate_event_order(events: list[Any], match_id: int) -> None:
        indices = [event.source_event_index for event in events]
        if len(indices) != len(set(indices)):
            raise RawDataFormatError(
                f"duplicate StatsBomb event index in match {match_id}"
            )
        if indices != sorted(indices):
            raise RawDataFormatError(
                f"StatsBomb event indexes are not ordered in match {match_id}"
            )

    @staticmethod
    def _lineup_interval_count(payload: dict[str, Any]) -> int:
        players = payload.get("lineup")
        if not isinstance(players, list):
            raise RawDataFormatError("lineup.lineup must be an array")
        count = 0
        for player in players:
            if not isinstance(player, dict):
                raise RawDataFormatError("lineup player must be an object")
            positions = player.get("positions")
            if not isinstance(positions, list):
                raise RawDataFormatError("lineup player positions must be an array")
            count += len(positions)
        return count

    def _upsert_dimensions(self, bundle: RawMatchBundle) -> None:
        match = bundle.match.payload
        home_team = self._object(match, "home_team", bundle)
        away_team = self._object(match, "away_team", bundle)
        competition = self._object(match, "competition", bundle)
        season = self._object(match, "season", bundle)

        home_team_id = self._positive_int(home_team.get("home_team_id"), "home_team_id")
        away_team_id = self._positive_int(away_team.get("away_team_id"), "away_team_id")
        teams: list[tuple[int, str]] = [
            (home_team_id, self._text(home_team.get("home_team_name"), "home_team_name")),
            (away_team_id, self._text(away_team.get("away_team_name"), "away_team_name")),
        ]
        players: list[tuple[int, str, int, str | None]] = []

        for lineup_record in bundle.lineups:
            lineup = lineup_record.payload
            team_id = self._positive_int(lineup.get("team_id"), "lineup.team_id")
            teams.append((team_id, self._text(lineup.get("team_name"), "lineup.team_name")))
            lineup_players = lineup.get("lineup")
            if not isinstance(lineup_players, list):
                raise RawDataFormatError("lineup.lineup must be an array")
            for player in lineup_players:
                if not isinstance(player, dict):
                    raise RawDataFormatError("lineup player must be an object")
                position = self._first_position(player.get("positions"))
                players.append(
                    (
                        self._positive_int(player.get("player_id"), "player_id"),
                        self._text(player.get("player_name"), "player_name"),
                        team_id,
                        position,
                    )
                )

        self.writer.upsert_teams(teams)
        self.writer.upsert_players(players)
        self.writer.upsert_match(
            match_id=bundle.match.match_id,
            competition_id=self._positive_int(
                competition.get("competition_id"), "competition_id"
            ),
            competition=self._text(
                competition.get("competition_name"), "competition_name"
            ),
            season_id=self._positive_int(season.get("season_id"), "season_id"),
            season=self._text(season.get("season_name"), "season_name"),
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_score=self._nonnegative_int(match.get("home_score"), "home_score"),
            away_score=self._nonnegative_int(match.get("away_score"), "away_score"),
            match_date=date.fromisoformat(
                self._text(match.get("match_date"), "match_date")
            ),
        )

    @staticmethod
    def _object(payload: dict[str, Any], key: str, bundle: RawMatchBundle) -> dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise RawDataFormatError(
                f"{key} must be an object for match {bundle.match.match_id}"
            )
        return value

    @staticmethod
    def _first_position(value: Any) -> str | None:
        if not isinstance(value, list) or not value:
            return None
        first = value[0]
        if not isinstance(first, dict):
            return None
        position = first.get("position")
        return position if isinstance(position, str) and position else None

    @staticmethod
    def _positive_int(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RawDataFormatError(f"{field} must be a positive integer")
        return value

    @staticmethod
    def _nonnegative_int(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RawDataFormatError(f"{field} must be a nonnegative integer")
        return value

    @staticmethod
    def _text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise RawDataFormatError(f"{field} must be non-empty text")
        return value
