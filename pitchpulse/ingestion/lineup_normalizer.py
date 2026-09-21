"""Normalize StatsBomb lineup records."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .normalizer import NormalizationError, StatsBombEventNormalizer
from .reader import RawRecord


@dataclass(frozen=True, slots=True)
class CanonicalLineupInterval:
    """One match-specific player position interval."""

    source: str
    source_record_index: int
    match_id: int
    team_id: int
    player_id: int
    position_id: int
    position: str
    from_seconds: int
    to_seconds: int | None
    from_period: int
    to_period: int | None
    start_reason: str
    end_reason: str
    raw_details: dict[str, Any]


class StatsBombLineupNormalizer:
    """Convert lineup positions into playing intervals."""

    _CLOCK_PATTERN = re.compile(r"^(\d+):(\d{2})$")

    def normalize(self, record: RawRecord) -> tuple[CanonicalLineupInterval, ...]:
        if record.kind != "lineup":
            raise NormalizationError(
                "unexpected_record_kind",
                f"Expected a lineup record, got {record.kind!r}",
                record,
            )
        payload = record.payload
        team_id = StatsBombEventNormalizer._int(
            payload.get("team_id"), "team_id", record
        )
        players = payload.get("lineup")
        if not isinstance(players, list):
            raise NormalizationError(
                "malformed_lineup", "lineup must be an array", record
            )

        intervals: list[CanonicalLineupInterval] = []
        for player_index, player in enumerate(players):
            if not isinstance(player, dict):
                raise NormalizationError(
                    "malformed_lineup_player",
                    f"lineup[{player_index}] must be an object",
                    record,
                )
            player_id = StatsBombEventNormalizer._int(
                player.get("player_id"),
                f"lineup[{player_index}].player_id",
                record,
            )
            positions = player.get("positions")
            if not isinstance(positions, list):
                raise NormalizationError(
                    "malformed_positions",
                    f"lineup[{player_index}].positions must be an array",
                    record,
                )
            for position_index, position in enumerate(positions):
                field = f"lineup[{player_index}].positions[{position_index}]"
                if not isinstance(position, dict):
                    raise NormalizationError(
                        "malformed_position", f"{field} must be an object", record
                    )
                from_seconds = self._clock_seconds(
                    position.get("from"), f"{field}.from", record
                )
                if from_seconds is None:
                    raise NormalizationError(
                        "malformed_lineup_time",
                        f"{field}.from is required",
                        record,
                    )
                intervals.append(
                    CanonicalLineupInterval(
                        source=record.source,
                        source_record_index=record.source_record_index,
                        match_id=record.match_id,
                        team_id=team_id,
                        player_id=player_id,
                        position_id=StatsBombEventNormalizer._int(
                            position.get("position_id"),
                            f"{field}.position_id",
                            record,
                        ),
                        position=self._text(
                            position.get("position"), f"{field}.position", record
                        ),
                        from_seconds=from_seconds,
                        to_seconds=self._clock_seconds(
                            position.get("to"), f"{field}.to", record
                        ),
                        from_period=StatsBombEventNormalizer._int(
                            position.get("from_period"),
                            f"{field}.from_period",
                            record,
                        ),
                        to_period=self._optional_int(
                            position.get("to_period"), f"{field}.to_period", record
                        ),
                        start_reason=self._text(
                            position.get("start_reason"),
                            f"{field}.start_reason",
                            record,
                        ),
                        end_reason=self._text(
                            position.get("end_reason"),
                            f"{field}.end_reason",
                            record,
                        ),
                        raw_details=copy.deepcopy(position),
                    )
                )
        return tuple(intervals)

    def normalize_many(
        self, records: Iterable[RawRecord]
    ) -> Iterator[CanonicalLineupInterval]:
        for record in records:
            yield from self.normalize(record)

    def _clock_seconds(
        self, value: Any, field: str, record: RawRecord
    ) -> int | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise NormalizationError(
                "malformed_lineup_time", f"{field} must be MM:SS text", record
            )
        match = self._CLOCK_PATTERN.fullmatch(value)
        if match is None:
            raise NormalizationError(
                "malformed_lineup_time", f"{field} must be MM:SS text", record
            )
        minutes, seconds = (int(part) for part in match.groups())
        if seconds > 59:
            raise NormalizationError(
                "malformed_lineup_time",
                f"{field} seconds must be between 00 and 59",
                record,
            )
        return minutes * 60 + seconds

    @staticmethod
    def _optional_int(value: Any, field: str, record: RawRecord) -> int | None:
        if value is None:
            return None
        return StatsBombEventNormalizer._int(value, field, record)

    @staticmethod
    def _text(value: Any, field: str, record: RawRecord) -> str:
        if not isinstance(value, str) or not value:
            raise NormalizationError(
                "malformed_field", f"{field} must be non-empty text", record
            )
        return value


__all__ = ["CanonicalLineupInterval", "StatsBombLineupNormalizer"]
