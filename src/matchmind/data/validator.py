"""Validate canonical football events before database persistence."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import time
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID

from .normalizer import (
    Canonical360Frame,
    CanonicalEvent,
    CanonicalLineupInterval,
    EVENT_TYPE_MAP,
)
from .raw_reader import RawDataFormatError, StatsBombRawReader


CANONICAL_EVENT_TYPES = frozenset(EVENT_TYPE_MAP.values())

PLAYER_OPTIONAL_TYPES = frozenset(
    {
        "half_end",
        "half_start",
        "own_goal_for",
        "referee_ball_drop",
        "starting_xi",
        "tactical_shift",
    }
)

LOCATION_OPTIONAL_TYPES = frozenset(
    {
        "bad_behaviour",
        "half_end",
        "half_start",
        "injury_stoppage",
        "player_off",
        "player_on",
        "starting_xi",
        "substitution",
        "tactical_shift",
    }
)

ENDPOINT_REQUIRED_TYPES = frozenset({"pass", "carry", "shot"})
ENDPOINT_OPTIONAL_TYPES = frozenset({"goalkeeper"})

OUTCOME_REQUIRED_TYPES = frozenset(
    {
        "fifty_fifty",
        "ball_receipt",
        "dribble",
        "interception",
        "pass",
        "shot",
        "substitution",
    }
)
OUTCOME_OPTIONAL_TYPES = frozenset({"duel", "goalkeeper"})

PERIOD_START_MINUTES: dict[int, int] = {
    1: 0,
    2: 45,
    3: 90,
    4: 105,
    5: 120,
}

SNAKE_CASE_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    event: CanonicalEvent
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class EventValidationContext:
    """Known teams and players for each match in the selected dataset."""

    match_team_ids: Mapping[int, frozenset[int]]
    match_player_ids: Mapping[int, frozenset[int]]

    @classmethod
    def from_reader(cls, reader: StatsBombRawReader) -> EventValidationContext:
        match_team_ids: dict[int, frozenset[int]] = {}
        match_player_ids: dict[int, frozenset[int]] = {}

        for match_record in reader.iter_matches():
            match_id = match_record.match_id
            match = match_record.payload
            team_ids = {
                cls._nested_positive_int(
                    match, "home_team", "home_team_id", match_record.source_file
                ),
                cls._nested_positive_int(
                    match, "away_team", "away_team_id", match_record.source_file
                ),
            }
            player_ids: set[int] = set()

            for lineup_record in reader.read_lineups(match_id):
                lineup = lineup_record.payload
                team_id = cls._positive_int(
                    lineup.get("team_id"), "team_id", lineup_record.source_file
                )
                team_ids.add(team_id)
                players = lineup.get("lineup")
                if not isinstance(players, list):
                    raise RawDataFormatError(
                        f"lineup must be an array in {lineup_record.source_file}, "
                        f"record {lineup_record.source_record_index}"
                    )
                for index, player in enumerate(players):
                    if not isinstance(player, dict):
                        raise RawDataFormatError(
                            f"lineup player {index} must be an object in "
                            f"{lineup_record.source_file}"
                        )
                    player_ids.add(
                        cls._positive_int(
                            player.get("player_id"),
                            f"lineup[{index}].player_id",
                            lineup_record.source_file,
                        )
                    )

            match_team_ids[match_id] = frozenset(team_ids)
            match_player_ids[match_id] = frozenset(player_ids)

        return cls(
            match_team_ids=MappingProxyType(match_team_ids),
            match_player_ids=MappingProxyType(match_player_ids),
        )

    @staticmethod
    def _nested_positive_int(
        payload: dict[str, Any], object_key: str, value_key: str, source_file: Any
    ) -> int:
        nested = payload.get(object_key)
        if not isinstance(nested, dict):
            raise RawDataFormatError(
                f"{object_key} must be an object in {source_file}"
            )
        return EventValidationContext._positive_int(
            nested.get(value_key), f"{object_key}.{value_key}", source_file
        )

    @staticmethod
    def _positive_int(value: Any, field: str, source_file: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RawDataFormatError(
                f"{field} must be a positive integer in {source_file}, got {value!r}"
            )
        return value


class CanonicalEventValidator:
    """Apply canonical and StatsBomb-specific business validation rules."""

    def __init__(self, context: EventValidationContext | None = None) -> None:
        self.context = context

    def validate(self, event: CanonicalEvent) -> ValidationResult:
        issues: list[ValidationIssue] = []

        self._validate_identity(event, issues)
        self._validate_references(event, issues)
        self._validate_time(event, issues)
        self._validate_coordinates(event, issues)
        self._validate_player(event, issues)
        self._validate_outcome(event, issues)
        self._validate_xg(event, issues)
        self._validate_enrichment(event, issues)

        return ValidationResult(event=event, issues=tuple(issues))

    @staticmethod
    def _add(
        issues: list[ValidationIssue], code: str, field: str, message: str
    ) -> None:
        issues.append(ValidationIssue(code=code, field=field, message=message))

    def _validate_identity(
        self, event: CanonicalEvent, issues: list[ValidationIssue]
    ) -> None:
        if not isinstance(event.source, str) or not event.source.strip():
            self._add(issues, "source_required", "source", "source is required")
        if not isinstance(event.source_event_id, UUID):
            self._add(
                issues,
                "source_event_id_invalid",
                "source_event_id",
                "source_event_id must be a UUID",
            )
        for field in (
            "match_id",
            "team_id",
            "source_event_index",
            "possession_id",
            "possession_team_id",
        ):
            value = getattr(event, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                self._add(
                    issues,
                    f"{field}_invalid",
                    field,
                    f"{field} must be a positive integer",
                )
        if (
            isinstance(event.source_record_index, bool)
            or not isinstance(event.source_record_index, int)
            or event.source_record_index < 0
        ):
            self._add(
                issues,
                "source_record_index_invalid",
                "source_record_index",
                "source_record_index must be a nonnegative integer",
            )
        if event.player_id is not None and (
            isinstance(event.player_id, bool)
            or not isinstance(event.player_id, int)
            or event.player_id <= 0
        ):
            self._add(
                issues,
                "player_id_invalid",
                "player_id",
                "player_id must be a positive integer when present",
            )
        if event.event_type not in CANONICAL_EVENT_TYPES:
            self._add(
                issues,
                "event_type_unknown",
                "event_type",
                f"event_type {event.event_type!r} is not canonical",
            )

    def _validate_references(
        self, event: CanonicalEvent, issues: list[ValidationIssue]
    ) -> None:
        if self.context is None:
            return
        teams = self.context.match_team_ids.get(event.match_id)
        players = self.context.match_player_ids.get(event.match_id)
        if teams is None:
            self._add(
                issues,
                "match_not_found",
                "match_id",
                f"match {event.match_id} is not in the selected dataset",
            )
            return
        if event.team_id not in teams:
            self._add(
                issues,
                "team_not_in_match",
                "team_id",
                f"team {event.team_id} does not participate in match {event.match_id}",
            )
        if event.possession_team_id not in teams:
            self._add(
                issues,
                "possession_team_not_in_match",
                "possession_team_id",
                f"possession team {event.possession_team_id} does not participate "
                f"in match {event.match_id}",
            )
        if event.player_id is not None and (
            players is None or event.player_id not in players
        ):
            self._add(
                issues,
                "player_not_in_match_lineup",
                "player_id",
                f"player {event.player_id} is not in a lineup for match {event.match_id}",
            )
        if event.recipient_id is not None and (
            players is None or event.recipient_id not in players
        ):
            self._add(
                issues,
                "recipient_not_in_match_lineup",
                "recipient_id",
                f"recipient {event.recipient_id} is not in a lineup for match "
                f"{event.match_id}",
            )

    def _validate_time(
        self, event: CanonicalEvent, issues: list[ValidationIssue]
    ) -> None:
        valid_period = (
            isinstance(event.period, int)
            and not isinstance(event.period, bool)
            and event.period in PERIOD_START_MINUTES
        )
        if not valid_period:
            self._add(
                issues,
                "period_invalid",
                "period",
                "period must be one of 1, 2, 3, 4, or 5",
            )
        if not isinstance(event.timestamp, time):
            self._add(
                issues,
                "timestamp_invalid",
                "timestamp",
                "timestamp must be a period-local time",
            )
        valid_minute = (
            isinstance(event.minute, int)
            and not isinstance(event.minute, bool)
            and event.minute >= 0
        )
        if not valid_minute:
            self._add(
                issues,
                "minute_invalid",
                "minute",
                "minute must be a nonnegative integer",
            )
        valid_second = (
            isinstance(event.second, int)
            and not isinstance(event.second, bool)
            and 0 <= event.second <= 59
        )
        if not valid_second:
            self._add(
                issues,
                "second_invalid",
                "second",
                "second must be between 0 and 59",
            )

        if not (valid_period and isinstance(event.timestamp, time)):
            return
        if not (valid_minute and valid_second):
            return

        period_seconds = (
            event.timestamp.hour * 3600
            + event.timestamp.minute * 60
            + event.timestamp.second
        )
        match_seconds = PERIOD_START_MINUTES[event.period] * 60 + period_seconds
        expected_minute, expected_second = divmod(match_seconds, 60)
        if event.minute != expected_minute or event.second != expected_second:
            self._add(
                issues,
                "time_fields_inconsistent",
                "timestamp",
                "period/timestamp does not agree with minute/second: "
                f"expected {expected_minute}:{expected_second:02d}",
            )

    def _validate_coordinates(
        self, event: CanonicalEvent, issues: list[ValidationIssue]
    ) -> None:
        start_present = self._validate_coordinate_pair(
            event.x, event.y, "start", issues
        )
        endpoint_present = self._validate_coordinate_pair(
            event.end_x, event.end_y, "end", issues
        )

        if event.event_type in CANONICAL_EVENT_TYPES:
            if not start_present and event.event_type not in LOCATION_OPTIONAL_TYPES:
                self._add(
                    issues,
                    "start_location_required",
                    "x",
                    f"start location is required for {event.event_type}",
                )
            if event.event_type in ENDPOINT_REQUIRED_TYPES and not endpoint_present:
                self._add(
                    issues,
                    "endpoint_required",
                    "end_x",
                    f"endpoint is required for {event.event_type}",
                )
            if (
                event.event_type
                not in ENDPOINT_REQUIRED_TYPES | ENDPOINT_OPTIONAL_TYPES
                and endpoint_present
            ):
                self._add(
                    issues,
                    "endpoint_not_allowed",
                    "end_x",
                    f"endpoint is not defined for {event.event_type}",
                )

    def _validate_coordinate_pair(
        self,
        x: float | None,
        y: float | None,
        prefix: str,
        issues: list[ValidationIssue],
    ) -> bool:
        if (x is None) != (y is None):
            self._add(
                issues,
                f"{prefix}_coordinate_pair_incomplete",
                f"{prefix}_x",
                f"{prefix} x and y must both be present or both be null",
            )
            return False
        if x is None:
            return False
        for axis, value in (("x", x), ("y", y)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 100
            ):
                self._add(
                    issues,
                    f"{prefix}_{axis}_out_of_range",
                    f"{prefix}_{axis}",
                    f"{prefix} {axis} must be a finite number in 0..100",
                )
        return True

    def _validate_player(
        self, event: CanonicalEvent, issues: list[ValidationIssue]
    ) -> None:
        if (
            event.event_type in CANONICAL_EVENT_TYPES
            and event.player_id is None
            and event.event_type not in PLAYER_OPTIONAL_TYPES
        ):
            self._add(
                issues,
                "player_required",
                "player_id",
                f"player_id is required for {event.event_type}",
            )

    def _validate_outcome(
        self, event: CanonicalEvent, issues: list[ValidationIssue]
    ) -> None:
        if event.outcome is not None and (
            not isinstance(event.outcome, str)
            or not SNAKE_CASE_PATTERN.fullmatch(event.outcome)
        ):
            self._add(
                issues,
                "outcome_invalid",
                "outcome",
                "outcome must be canonical lowercase snake_case",
            )

        if event.event_type in OUTCOME_REQUIRED_TYPES and event.outcome is None:
            self._add(
                issues,
                "outcome_required",
                "outcome",
                f"outcome is required for {event.event_type}",
            )
        elif (
            event.event_type in CANONICAL_EVENT_TYPES
            and event.event_type
            not in OUTCOME_REQUIRED_TYPES | OUTCOME_OPTIONAL_TYPES
            and event.outcome is not None
        ):
            self._add(
                issues,
                "outcome_not_allowed",
                "outcome",
                f"outcome is not defined for {event.event_type}",
            )

    def _validate_xg(
        self, event: CanonicalEvent, issues: list[ValidationIssue]
    ) -> None:
        if event.event_type == "shot" and event.xg is None:
            self._add(
                issues, "xg_required", "xg", "xg is required for shot events"
            )
            return
        if event.event_type != "shot" and event.xg is not None:
            self._add(
                issues,
                "xg_not_allowed",
                "xg",
                "xg must be null for non-shot events",
            )
            return
        if event.xg is not None and (
            isinstance(event.xg, bool)
            or not isinstance(event.xg, Real)
            or not math.isfinite(float(event.xg))
            or not 0 <= float(event.xg) <= 1
        ):
            self._add(
                issues,
                "xg_out_of_range",
                "xg",
                "xg must be a finite number in 0..1",
            )

    def _validate_enrichment(
        self, event: CanonicalEvent, issues: list[ValidationIssue]
    ) -> None:
        if event.duration is not None and (
            isinstance(event.duration, bool)
            or not isinstance(event.duration, Real)
            or not math.isfinite(float(event.duration))
            or event.duration < 0
        ):
            self._add(
                issues,
                "duration_invalid",
                "duration",
                "duration must be a finite nonnegative number when present",
            )
        for field in ("event_subtype", "body_part", "play_pattern"):
            value = getattr(event, field)
            if value is not None and (
                not isinstance(value, str) or not SNAKE_CASE_PATTERN.fullmatch(value)
            ):
                self._add(
                    issues,
                    f"{field}_invalid",
                    field,
                    f"{field} must be canonical lowercase snake_case",
                )
        if not event.play_pattern:
            self._add(
                issues,
                "play_pattern_required",
                "play_pattern",
                "play_pattern is required",
            )
        for field in ("under_pressure", "counterpress"):
            if not isinstance(getattr(event, field), bool):
                self._add(
                    issues,
                    f"{field}_invalid",
                    field,
                    f"{field} must be boolean",
                )
        if event.recipient_id is not None and (
            isinstance(event.recipient_id, bool)
            or not isinstance(event.recipient_id, int)
            or event.recipient_id <= 0
        ):
            self._add(
                issues,
                "recipient_id_invalid",
                "recipient_id",
                "recipient_id must be a positive integer when present",
            )
        if not isinstance(event.related_event_ids, tuple) or any(
            not isinstance(value, UUID) for value in event.related_event_ids
        ):
            self._add(
                issues,
                "related_event_ids_invalid",
                "related_event_ids",
                "related_event_ids must contain UUID values",
            )
        if not isinstance(event.raw_details, dict):
            self._add(
                issues,
                "raw_details_invalid",
                "raw_details",
                "raw_details must be an object",
            )


@dataclass(frozen=True, slots=True)
class LineupIntervalValidationResult:
    interval: CanonicalLineupInterval
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


class CanonicalLineupIntervalValidator:
    """Validate match-specific playing intervals before persistence."""

    def validate(
        self, interval: CanonicalLineupInterval
    ) -> LineupIntervalValidationResult:
        issues: list[ValidationIssue] = []
        if interval.from_period not in PERIOD_START_MINUTES:
            issues.append(
                ValidationIssue(
                    code="lineup_from_period_invalid",
                    field="from_period",
                    message="from_period must be between 1 and 5",
                )
            )
        if (
            interval.to_period is not None
            and interval.to_period not in PERIOD_START_MINUTES
        ):
            issues.append(
                ValidationIssue(
                    code="lineup_to_period_invalid",
                    field="to_period",
                    message="to_period must be between 1 and 5",
                )
            )
        return LineupIntervalValidationResult(interval, tuple(issues))


@dataclass(frozen=True, slots=True)
class ThreeSixtyValidationResult:
    frame: Canonical360Frame
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


class Canonical360Validator:
    """Validate StatsBomb 360 geometry and its event link."""

    def validate(
        self,
        frame: Canonical360Frame,
        *,
        known_event_ids: set[UUID] | frozenset[UUID] | None = None,
    ) -> ThreeSixtyValidationResult:
        issues: list[ValidationIssue] = []
        if known_event_ids is not None and frame.source_event_id not in known_event_ids:
            issues.append(
                ValidationIssue(
                    code="three_sixty_event_not_found",
                    field="source_event_id",
                    message="360 event UUID does not link to a valid event in the match",
                )
            )
        if len(frame.visible_area) < 6 or len(frame.visible_area) % 2:
            issues.append(
                ValidationIssue(
                    code="visible_area_invalid",
                    field="visible_area",
                    message="visible_area must contain at least three x/y pairs",
                )
            )
        else:
            self._validate_flat_coordinates(
                frame.visible_area, "visible_area", issues
            )

        actor_count = 0
        for index, player in enumerate(frame.freeze_frame):
            prefix = f"freeze_frame[{index}]"
            for field in ("teammate", "actor", "keeper"):
                if not isinstance(player.get(field), bool):
                    issues.append(
                        ValidationIssue(
                            code="freeze_frame_flag_invalid",
                            field=f"{prefix}.{field}",
                            message=f"{prefix}.{field} must be boolean",
                        )
                    )
            if player.get("actor") is True:
                actor_count += 1
            location = player.get("location")
            if not isinstance(location, list) or len(location) < 2:
                issues.append(
                    ValidationIssue(
                        code="freeze_frame_location_invalid",
                        field=f"{prefix}.location",
                        message=f"{prefix}.location must contain x and y",
                    )
                )
                continue
            self._validate_xy(location[0], location[1], f"{prefix}.location", issues)
        if actor_count > 1:
            issues.append(
                ValidationIssue(
                    code="freeze_frame_multiple_actors",
                    field="freeze_frame",
                    message="freeze_frame cannot contain more than one actor",
                )
            )
        return ThreeSixtyValidationResult(frame, tuple(issues))

    def _validate_flat_coordinates(
        self,
        values: tuple[float, ...],
        field: str,
        issues: list[ValidationIssue],
    ) -> None:
        for index in range(0, len(values), 2):
            self._validate_xy(values[index], values[index + 1], field, issues)

    @staticmethod
    def _validate_xy(
        x: Any, y: Any, field: str, issues: list[ValidationIssue]
    ) -> None:
        valid_x = (
            not isinstance(x, bool)
            and isinstance(x, Real)
            and math.isfinite(float(x))
        )
        valid_y = (
            not isinstance(y, bool)
            and isinstance(y, Real)
            and math.isfinite(float(y))
        )
        if not (valid_x and valid_y):
            issues.append(
                ValidationIssue(
                    code="three_sixty_coordinate_invalid",
                    field=field,
                    message="360 coordinates must be finite numeric values",
                )
            )
