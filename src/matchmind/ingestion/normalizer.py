"""Normalize raw StatsBomb event records into the canonical event contract."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import time
from typing import Any
from uuid import UUID

from .reader import RawRecord


EVENT_TYPE_MAP: dict[str, str] = {
    "50/50": "fifty_fifty",
    "Bad Behaviour": "bad_behaviour",
    "Ball Receipt*": "ball_receipt",
    "Ball Recovery": "ball_recovery",
    "Block": "block",
    "Carry": "carry",
    "Clearance": "clearance",
    "Dispossessed": "dispossessed",
    "Dribble": "dribble",
    "Dribbled Past": "dribbled_past",
    "Duel": "duel",
    "Error": "error",
    "Foul Committed": "foul_committed",
    "Foul Won": "foul_won",
    "Goal Keeper": "goalkeeper",
    "Half End": "half_end",
    "Half Start": "half_start",
    "Injury Stoppage": "injury_stoppage",
    "Interception": "interception",
    "Miscontrol": "miscontrol",
    "Offside": "offside",
    "Own Goal Against": "own_goal_against",
    "Own Goal For": "own_goal_for",
    "Pass": "pass",
    "Player Off": "player_off",
    "Player On": "player_on",
    "Pressure": "pressure",
    "Referee Ball-Drop": "referee_ball_drop",
    "Shield": "shield",
    "Shot": "shot",
    "Starting XI": "starting_xi",
    "Substitution": "substitution",
    "Tactical Shift": "tactical_shift",
}

ENDPOINT_DETAIL_KEYS: dict[str, str] = {
    "pass": "pass",
    "carry": "carry",
    "shot": "shot",
    "goalkeeper": "goalkeeper",
}

EVENT_DETAIL_KEYS: dict[str, str] = {
    "fifty_fifty": "50_50",
    "ball_receipt": "ball_receipt",
    "ball_recovery": "ball_recovery",
    "block": "block",
    "carry": "carry",
    "clearance": "clearance",
    "dribble": "dribble",
    "duel": "duel",
    "foul_committed": "foul_committed",
    "foul_won": "foul_won",
    "goalkeeper": "goalkeeper",
    "injury_stoppage": "injury_stoppage",
    "interception": "interception",
    "pass": "pass",
    "player_off": "player_off",
    "shot": "shot",
    "substitution": "substitution",
}

OUTCOME_DETAIL_KEYS: dict[str, str] = {
    "fifty_fifty": "50_50",
    "ball_receipt": "ball_receipt",
    "dribble": "dribble",
    "duel": "duel",
    "goalkeeper": "goalkeeper",
    "interception": "interception",
    "pass": "pass",
    "shot": "shot",
    "substitution": "substitution",
}

IMPLICIT_COMPLETE_TYPES = {"ball_receipt", "pass"}


class NormalizationError(Exception):
    """A raw event cannot be converted into the canonical field types."""

    def __init__(self, code: str, message: str, record: RawRecord) -> None:
        self.code = code
        self.message = message
        self.record = record
        super().__init__(
            f"{code}: {message} "
            f"({record.source_file}, record {record.source_record_index})"
        )


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """A normalized event ready for business validation."""

    source: str
    source_event_id: UUID
    source_record_index: int
    source_event_index: int
    match_id: int
    team_id: int
    player_id: int | None
    possession_id: int
    possession_team_id: int
    event_type: str
    event_subtype: str | None
    period: int
    timestamp: time
    minute: int
    second: int
    duration: float | None
    x: float | None
    y: float | None
    end_x: float | None
    end_y: float | None
    outcome: str | None
    body_part: str | None
    recipient_id: int | None
    xg: float | None
    play_pattern: str
    under_pressure: bool
    counterpress: bool
    related_event_ids: tuple[UUID, ...]
    raw_details: dict[str, Any]

    def as_serializable_dict(self) -> dict[str, Any]:
        """Return JSON-friendly values for inspection and report output."""
        return {
            "source": self.source,
            "source_event_id": str(self.source_event_id),
            "source_record_index": self.source_record_index,
            "source_event_index": self.source_event_index,
            "match_id": self.match_id,
            "team_id": self.team_id,
            "player_id": self.player_id,
            "possession_id": self.possession_id,
            "possession_team_id": self.possession_team_id,
            "event_type": self.event_type,
            "event_subtype": self.event_subtype,
            "period": self.period,
            "timestamp": self.timestamp.isoformat(timespec="milliseconds"),
            "minute": self.minute,
            "second": self.second,
            "duration": self.duration,
            "x": self.x,
            "y": self.y,
            "end_x": self.end_x,
            "end_y": self.end_y,
            "outcome": self.outcome,
            "body_part": self.body_part,
            "recipient_id": self.recipient_id,
            "xg": self.xg,
            "play_pattern": self.play_pattern,
            "under_pressure": self.under_pressure,
            "counterpress": self.counterpress,
            "related_event_ids": [str(value) for value in self.related_event_ids],
            "raw_details": self.raw_details,
        }


class StatsBombEventNormalizer:
    """Convert StatsBomb event payloads into canonical events."""

    def normalize(self, record: RawRecord) -> CanonicalEvent:
        if record.kind != "event":
            raise NormalizationError(
                "unexpected_record_kind",
                f"Expected an event record, got {record.kind!r}",
                record,
            )

        payload = record.payload
        source_event_id = self._uuid(payload.get("id"), "id", record)
        source_event_index = self._int(payload.get("index"), "index", record)
        team_id = self._nested_int(payload, "team", "id", record, required=True)
        player_id = self._nested_int(
            payload, "player", "id", record, required=False
        )
        possession_id = self._int(payload.get("possession"), "possession", record)
        possession_team_id = self._nested_int(
            payload, "possession_team", "id", record, required=True
        )
        source_event_type = self._nested_text(
            payload, "type", "name", record, required=True
        )
        event_type = EVENT_TYPE_MAP.get(source_event_type)
        if event_type is None:
            raise NormalizationError(
                "unknown_event_type",
                f"StatsBomb event type {source_event_type!r} is not mapped",
                record,
            )

        detail = self._event_detail(payload, event_type, record)
        event_subtype = self._optional_nested_name(
            detail, "type", f"{EVENT_DETAIL_KEYS.get(event_type, event_type)}.type", record
        )
        body_part = self._optional_nested_name(
            detail,
            "body_part",
            f"{EVENT_DETAIL_KEYS.get(event_type, event_type)}.body_part",
            record,
        )
        recipient_id = self._optional_nested_id(
            detail,
            "recipient",
            f"{EVENT_DETAIL_KEYS.get(event_type, event_type)}.recipient",
            record,
        )

        period = self._int(payload.get("period"), "period", record)
        timestamp = self._time(payload.get("timestamp"), record)
        minute = self._int(payload.get("minute"), "minute", record)
        second = self._int(payload.get("second"), "second", record)
        duration = self._optional_float(payload.get("duration"), "duration", record)
        x, y = self._coordinates(payload.get("location"), "location", record)
        end_x, end_y = self._endpoint(payload, event_type, record)
        outcome = self._outcome(payload, event_type, record)
        xg = self._xg(payload, event_type, record)
        play_pattern = self._canonical_nested_name(
            payload, "play_pattern", "name", record, required=True
        )
        under_pressure = self._optional_bool(
            payload.get("under_pressure"), "under_pressure", record
        )
        counterpress = self._optional_bool(
            payload.get("counterpress"), "counterpress", record
        )
        related_event_ids = self._related_event_ids(
            payload.get("related_events"), record
        )

        return CanonicalEvent(
            source=record.source,
            source_event_id=source_event_id,
            source_record_index=record.source_record_index,
            source_event_index=source_event_index,
            match_id=record.match_id,
            team_id=team_id,
            player_id=player_id,
            possession_id=possession_id,
            possession_team_id=possession_team_id,
            event_type=event_type,
            event_subtype=event_subtype,
            period=period,
            timestamp=timestamp,
            minute=minute,
            second=second,
            duration=duration,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            outcome=outcome,
            body_part=body_part,
            recipient_id=recipient_id,
            xg=xg,
            play_pattern=play_pattern,
            under_pressure=under_pressure,
            counterpress=counterpress,
            related_event_ids=related_event_ids,
            raw_details=copy.deepcopy(payload),
        )

    def normalize_many(self, records: Iterable[RawRecord]) -> Iterator[CanonicalEvent]:
        """Normalize records lazily and stop at the first normalization error."""
        for record in records:
            yield self.normalize(record)

    def _event_detail(
        self, payload: dict[str, Any], event_type: str, record: RawRecord
    ) -> dict[str, Any] | None:
        detail_key = EVENT_DETAIL_KEYS.get(event_type)
        if detail_key is None:
            return None
        detail = payload.get(detail_key)
        if detail is None:
            return None
        if not isinstance(detail, dict):
            raise NormalizationError(
                "malformed_event_detail",
                f"{detail_key} must be an object",
                record,
            )
        return detail

    def _optional_nested_name(
        self,
        parent: dict[str, Any] | None,
        key: str,
        field: str,
        record: RawRecord,
    ) -> str | None:
        if parent is None or parent.get(key) is None:
            return None
        value = parent.get(key)
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise NormalizationError(
                "malformed_field", f"{field}.name must be text", record
            )
        return self._canonical_text(value["name"], f"{field}.name", record)

    def _optional_nested_id(
        self,
        parent: dict[str, Any] | None,
        key: str,
        field: str,
        record: RawRecord,
    ) -> int | None:
        if parent is None or parent.get(key) is None:
            return None
        value = parent.get(key)
        if not isinstance(value, dict):
            raise NormalizationError(
                "malformed_field", f"{field} must be an object", record
            )
        return self._int(value.get("id"), f"{field}.id", record)

    def _canonical_nested_name(
        self,
        payload: dict[str, Any],
        object_key: str,
        value_key: str,
        record: RawRecord,
        *,
        required: bool,
    ) -> str:
        value = self._nested_text(
            payload, object_key, value_key, record, required=required
        )
        return self._canonical_text(
            value, f"{object_key}.{value_key}", record
        )

    @staticmethod
    def _canonical_text(value: str, field: str, record: RawRecord) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
        if not normalized:
            raise NormalizationError(
                "malformed_field", f"{field} cannot be empty", record
            )
        return normalized

    def _related_event_ids(
        self, value: Any, record: RawRecord
    ) -> tuple[UUID, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise NormalizationError(
                "malformed_related_events",
                "related_events must be an array",
                record,
            )
        return tuple(
            self._uuid(item, f"related_events[{index}]", record)
            for index, item in enumerate(value)
        )

    def _optional_float(
        self, value: Any, field: str, record: RawRecord
    ) -> float | None:
        if value is None:
            return None
        return self._float(value, field, record)

    @staticmethod
    def _optional_bool(value: Any, field: str, record: RawRecord) -> bool:
        if value is None:
            return False
        if not isinstance(value, bool):
            raise NormalizationError(
                "malformed_boolean", f"{field} must be boolean", record
            )
        return value

    def _endpoint(
        self, payload: dict[str, Any], event_type: str, record: RawRecord
    ) -> tuple[float | None, float | None]:
        detail_key = ENDPOINT_DETAIL_KEYS.get(event_type)
        if detail_key is None:
            return None, None
        detail = payload.get(detail_key)
        if detail is None:
            return None, None
        if not isinstance(detail, dict):
            raise NormalizationError(
                "malformed_event_detail",
                f"{detail_key} must be an object",
                record,
            )
        return self._coordinates(
            detail.get("end_location"), f"{detail_key}.end_location", record
        )

    def _outcome(
        self, payload: dict[str, Any], event_type: str, record: RawRecord
    ) -> str | None:
        detail_key = OUTCOME_DETAIL_KEYS.get(event_type)
        if detail_key is None:
            return None

        detail = payload.get(detail_key)
        if detail is None:
            return "complete" if event_type in IMPLICIT_COMPLETE_TYPES else None
        if not isinstance(detail, dict):
            raise NormalizationError(
                "malformed_event_detail",
                f"{detail_key} must be an object",
                record,
            )

        outcome = detail.get("outcome")
        if outcome is None:
            return "complete" if event_type in IMPLICIT_COMPLETE_TYPES else None
        if not isinstance(outcome, dict) or not isinstance(outcome.get("name"), str):
            raise NormalizationError(
                "malformed_outcome",
                f"{detail_key}.outcome.name must be text",
                record,
            )

        normalized = re.sub(r"[^a-z0-9]+", "_", outcome["name"].casefold()).strip(
            "_"
        )
        if not normalized:
            raise NormalizationError(
                "malformed_outcome",
                f"{detail_key}.outcome.name cannot be empty",
                record,
            )
        return normalized

    def _xg(
        self, payload: dict[str, Any], event_type: str, record: RawRecord
    ) -> float | None:
        if event_type != "shot":
            return None
        shot = payload.get("shot")
        if shot is None:
            return None
        if not isinstance(shot, dict):
            raise NormalizationError(
                "malformed_event_detail", "shot must be an object", record
            )
        value = shot.get("statsbomb_xg")
        if value is None:
            return None
        return self._float(value, "shot.statsbomb_xg", record)

    def _coordinates(
        self, value: Any, field: str, record: RawRecord
    ) -> tuple[float | None, float | None]:
        if value is None:
            return None, None
        if not isinstance(value, list) or len(value) < 2:
            raise NormalizationError(
                "malformed_coordinates",
                f"{field} must contain at least x and y",
                record,
            )
        source_x = self._float(value[0], f"{field}[0]", record)
        source_y = self._float(value[1], f"{field}[1]", record)
        return source_x / 120 * 100, source_y / 80 * 100

    @staticmethod
    def _nested_int(
        payload: dict[str, Any],
        object_key: str,
        value_key: str,
        record: RawRecord,
        *,
        required: bool,
    ) -> int | None:
        nested = payload.get(object_key)
        if nested is None and not required:
            return None
        if not isinstance(nested, dict):
            raise NormalizationError(
                "missing_required_field" if required else "malformed_field",
                f"{object_key} must be an object",
                record,
            )
        value = nested.get(value_key)
        if value is None and not required:
            return None
        return StatsBombEventNormalizer._int(
            value, f"{object_key}.{value_key}", record
        )

    @staticmethod
    def _nested_text(
        payload: dict[str, Any],
        object_key: str,
        value_key: str,
        record: RawRecord,
        *,
        required: bool,
    ) -> str:
        nested = payload.get(object_key)
        if not isinstance(nested, dict):
            raise NormalizationError(
                "missing_required_field" if required else "malformed_field",
                f"{object_key} must be an object",
                record,
            )
        value = nested.get(value_key)
        if not isinstance(value, str) or not value:
            raise NormalizationError(
                "missing_required_field" if required else "malformed_field",
                f"{object_key}.{value_key} must be non-empty text",
                record,
            )
        return value

    @staticmethod
    def _uuid(value: Any, field: str, record: RawRecord) -> UUID:
        if not isinstance(value, str):
            raise NormalizationError(
                "malformed_uuid", f"{field} must be a UUID string", record
            )
        try:
            return UUID(value)
        except ValueError as exc:
            raise NormalizationError(
                "malformed_uuid", f"{field} is not a valid UUID: {value!r}", record
            ) from exc

    @staticmethod
    def _int(value: Any, field: str, record: RawRecord) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise NormalizationError(
                "malformed_integer", f"{field} must be an integer", record
            )
        return value

    @staticmethod
    def _float(value: Any, field: str, record: RawRecord) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NormalizationError(
                "malformed_number", f"{field} must be numeric", record
            )
        return float(value)

    @staticmethod
    def _time(value: Any, record: RawRecord) -> time:
        if not isinstance(value, str):
            raise NormalizationError(
                "malformed_timestamp", "timestamp must be text", record
            )
        try:
            return time.fromisoformat(value)
        except ValueError as exc:
            raise NormalizationError(
                "malformed_timestamp",
                f"timestamp is not HH:MM:SS.sss: {value!r}",
                record,
            ) from exc


@dataclass(frozen=True, slots=True)
class CanonicalLineupInterval:
    """One match-specific player position interval from a lineup file."""

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
    """Convert StatsBomb lineup position records into playing intervals."""

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
                position_id = StatsBombEventNormalizer._int(
                    position.get("position_id"), f"{field}.position_id", record
                )
                position_name = self._text(
                    position.get("position"), f"{field}.position", record
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
                to_seconds = self._clock_seconds(
                    position.get("to"), f"{field}.to", record
                )
                intervals.append(
                    CanonicalLineupInterval(
                        source=record.source,
                        source_record_index=record.source_record_index,
                        match_id=record.match_id,
                        team_id=team_id,
                        player_id=player_id,
                        position_id=position_id,
                        position=position_name,
                        from_seconds=from_seconds,
                        to_seconds=to_seconds,
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


@dataclass(frozen=True, slots=True)
class Canonical360Frame:
    """One StatsBomb 360 snapshot linked to an event UUID."""

    source: str
    source_event_id: UUID
    source_record_index: int
    match_id: int
    visible_area: tuple[float, ...]
    freeze_frame: tuple[dict[str, Any], ...]
    raw_details: dict[str, Any]

    def as_serializable_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_event_id": str(self.source_event_id),
            "source_record_index": self.source_record_index,
            "match_id": self.match_id,
            "visible_area": list(self.visible_area),
            "freeze_frame": list(self.freeze_frame),
        }


class StatsBomb360Normalizer:
    """Normalize the outer shape of StatsBomb 360 records without losing detail."""

    def normalize(self, record: RawRecord) -> Canonical360Frame:
        if record.kind != "three_sixty":
            raise NormalizationError(
                "unexpected_record_kind",
                f"Expected a three_sixty record, got {record.kind!r}",
                record,
            )
        payload = record.payload
        event_id = StatsBombEventNormalizer._uuid(
            payload.get("event_uuid"), "event_uuid", record
        )
        visible_area_value = payload.get("visible_area")
        if not isinstance(visible_area_value, list):
            raise NormalizationError(
                "malformed_visible_area", "visible_area must be an array", record
            )
        visible_area = tuple(
            StatsBombEventNormalizer._float(
                value, f"visible_area[{index}]", record
            )
            for index, value in enumerate(visible_area_value)
        )

        freeze_frame_value = payload.get("freeze_frame")
        if not isinstance(freeze_frame_value, list):
            raise NormalizationError(
                "malformed_freeze_frame", "freeze_frame must be an array", record
            )
        freeze_frame: list[dict[str, Any]] = []
        for index, player in enumerate(freeze_frame_value):
            if not isinstance(player, dict):
                raise NormalizationError(
                    "malformed_freeze_frame_player",
                    f"freeze_frame[{index}] must be an object",
                    record,
                )
            freeze_frame.append(copy.deepcopy(player))

        return Canonical360Frame(
            source=record.source,
            source_event_id=event_id,
            source_record_index=record.source_record_index,
            match_id=record.match_id,
            visible_area=visible_area,
            freeze_frame=tuple(freeze_frame),
            raw_details=copy.deepcopy(payload),
        )

    def normalize_many(
        self, records: Iterable[RawRecord]
    ) -> Iterator[Canonical360Frame]:
        for record in records:
            yield self.normalize(record)
