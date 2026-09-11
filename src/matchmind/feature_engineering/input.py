"""Validated source boundary for deterministic SPADL conversion.

The ingestion validators check individual records. This module checks the
cross-record guarantees that deterministic action conversion relies on,
independently of storage technology.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import time
from typing import Any
from uuid import UUID

from matchmind.ingestion.normalizer import (
    Canonical360Frame,
    CanonicalEvent,
    CanonicalLineupInterval,
    EVENT_DETAIL_KEYS,
)

SNAKE_CASE_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
POSSESSION_ORDER_EXEMPT_TYPES = frozenset({"ball_receipt"})
SOCCERACTION_ACTIONABLE_EVENT_TYPES = frozenset(
    {
        "pass",
        "dribble",
        "carry",
        "foul_committed",
        "duel",
        "interception",
        "shot",
        "own_goal_against",
        "goalkeeper",
        "clearance",
        "miscontrol",
    }
)

# udt_name is used instead of information_schema.data_type because it
# distinguishes int2/int4/int8 and exposes uuid[] as _uuid.


@dataclass(frozen=True, slots=True)
class ContractIssue:
    """One actionable violation of the Plan 02 -> Plan 03 boundary."""

    code: str
    entity: str
    message: str
    match_id: int | None = None
    source_event_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class SpadlInputValidationReport:
    """Validation counts and violations suitable for CLI/JSON reporting."""

    match_count: int
    event_count: int
    lineup_interval_count: int
    three_sixty_count: int
    events_with_subtype: int
    issues: tuple[ContractIssue, ...]
    warnings: tuple[ContractIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            raise SpadlInputContractError(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "match_count": self.match_count,
            "event_count": self.event_count,
            "lineup_interval_count": self.lineup_interval_count,
            "three_sixty_count": self.three_sixty_count,
            "events_with_subtype": self.events_with_subtype,
            "issues": [
                {
                    "code": issue.code,
                    "entity": issue.entity,
                    "message": issue.message,
                    "match_id": issue.match_id,
                    "source_event_id": (
                        str(issue.source_event_id)
                        if issue.source_event_id is not None
                        else None
                    ),
                }
                for issue in self.issues
            ],
            "warnings": [
                {
                    "code": warning.code,
                    "entity": warning.entity,
                    "message": warning.message,
                    "match_id": warning.match_id,
                    "source_event_id": (
                        str(warning.source_event_id)
                        if warning.source_event_id is not None
                        else None
                    ),
                }
                for warning in self.warnings
            ],
        }


class SpadlInputContractError(RuntimeError):
    """Raised before conversion when the Plan 02 contract is not usable."""

    def __init__(self, report: SpadlInputValidationReport) -> None:
        self.report = report
        preview = "; ".join(
            f"{issue.code}: {issue.message}" for issue in report.issues[:3]
        )
        remainder = len(report.issues) - 3
        if remainder > 0:
            preview += f"; and {remainder} more"
        super().__init__(f"SPADL input contract failed: {preview}")


@dataclass(frozen=True, slots=True)
class SpadlInput:
    """Validated source records and match context consumed by SPADL conversion."""

    events: tuple[CanonicalEvent, ...]
    lineup_intervals: tuple[CanonicalLineupInterval, ...]
    three_sixty_by_event: dict[tuple[str, UUID], Canonical360Frame]
    home_team_by_match: dict[int, int]
    away_team_by_match: dict[int, int]
    report: SpadlInputValidationReport


class SpadlInputValidator:
    """Validate cross-record ordering, possession, lineup, subtype and 360 links."""

    def validate(
        self,
        events: Sequence[CanonicalEvent],
        lineup_intervals: Sequence[CanonicalLineupInterval],
        three_sixty: Sequence[Canonical360Frame],
    ) -> SpadlInputValidationReport:
        issues: list[ContractIssue] = []
        warnings: list[ContractIssue] = []
        self._validate_events(events, issues)
        self._validate_possessions(events, issues)
        self._validate_lineups(events, lineup_intervals, issues, warnings)
        self._validate_three_sixty(events, three_sixty, issues)
        return SpadlInputValidationReport(
            match_count=len({event.match_id for event in events}),
            event_count=len(events),
            lineup_interval_count=len(lineup_intervals),
            three_sixty_count=len(three_sixty),
            events_with_subtype=sum(
                event.event_subtype is not None for event in events
            ),
            issues=tuple(issues),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _issue(
        issues: list[ContractIssue],
        code: str,
        entity: str,
        message: str,
        *,
        match_id: int | None = None,
        source_event_id: UUID | None = None,
    ) -> None:
        issues.append(
            ContractIssue(code, entity, message, match_id, source_event_id)
        )

    def _validate_events(
        self,
        events: Sequence[CanonicalEvent],
        issues: list[ContractIssue],
    ) -> None:
        if not events:
            self._issue(
                issues,
                "events_empty",
                "events",
                "at least one enriched event is required",
            )
            return

        seen_ids: set[tuple[str, UUID]] = set()
        previous_index: dict[int, int] = {}
        seen_order: set[tuple[int, int]] = set()
        known_events = {
            (event.source, event.source_event_id): event.match_id for event in events
        }

        for event in events:
            event_key = (event.source, event.source_event_id)
            if event_key in seen_ids:
                self._issue(
                    issues,
                    "duplicate_source_event_id",
                    "events",
                    "source_event_id must be unique within a source",
                    match_id=event.match_id,
                    source_event_id=event.source_event_id,
                )
            seen_ids.add(event_key)

            if (
                isinstance(event.source_event_index, bool)
                or not isinstance(event.source_event_index, int)
                or event.source_event_index <= 0
            ):
                self._issue(
                    issues,
                    "source_event_index_invalid",
                    "events",
                    "source_event_index must be a positive integer",
                    match_id=event.match_id,
                    source_event_id=event.source_event_id,
                )
            else:
                order_key = (event.match_id, event.source_event_index)
                if order_key in seen_order:
                    self._issue(
                        issues,
                        "duplicate_source_event_index",
                        "events",
                        "source_event_index must be unique within a match",
                        match_id=event.match_id,
                        source_event_id=event.source_event_id,
                    )
                seen_order.add(order_key)
                prior = previous_index.get(event.match_id)
                if prior is not None and event.source_event_index <= prior:
                    self._issue(
                        issues,
                        "source_event_order_invalid",
                        "events",
                        "events must be supplied in strictly increasing "
                        "source_event_index order within each match",
                        match_id=event.match_id,
                        source_event_id=event.source_event_id,
                    )
                previous_index[event.match_id] = event.source_event_index

            self._validate_event_fields(event, issues)
            related_ids = event.related_event_ids
            if not isinstance(related_ids, (tuple, list)):
                continue
            for related_id in related_ids:
                related_match = known_events.get((event.source, related_id))
                if related_match is None:
                    self._issue(
                        issues,
                        "related_event_not_found",
                        "events",
                        f"related event {related_id} is missing from the input",
                        match_id=event.match_id,
                        source_event_id=event.source_event_id,
                    )
                elif related_match != event.match_id:
                    self._issue(
                        issues,
                        "related_event_match_mismatch",
                        "events",
                        f"related event {related_id} belongs to match "
                        f"{related_match}",
                        match_id=event.match_id,
                        source_event_id=event.source_event_id,
                    )

    def _validate_event_fields(
        self, event: CanonicalEvent, issues: list[ContractIssue]
    ) -> None:
        required = {
            "source": event.source,
            "source_event_id": event.source_event_id,
            "source_record_index": event.source_record_index,
            "match_id": event.match_id,
            "team_id": event.team_id,
            "possession_id": event.possession_id,
            "possession_team_id": event.possession_team_id,
            "event_type": event.event_type,
            "period": event.period,
            "timestamp": event.timestamp,
            "minute": event.minute,
            "second": event.second,
            "play_pattern": event.play_pattern,
            "under_pressure": event.under_pressure,
            "counterpress": event.counterpress,
            "related_event_ids": event.related_event_ids,
            "raw_details": event.raw_details,
        }
        for field, value in required.items():
            if value is None:
                self._issue(
                    issues,
                    "required_event_field_missing",
                    "events",
                    f"required field {field} is null",
                    match_id=event.match_id,
                    source_event_id=event.source_event_id,
                )

        if not isinstance(event.source_event_id, UUID):
            self._issue(
                issues,
                "source_event_id_invalid",
                "events",
                "source_event_id must be a UUID",
                match_id=event.match_id,
            )
        for field in (
            "source_record_index",
            "match_id",
            "team_id",
            "possession_id",
            "possession_team_id",
            "period",
            "minute",
            "second",
        ):
            value = getattr(event, field)
            minimum = 0 if field in {"source_record_index", "minute", "second"} else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                self._issue(
                    issues,
                    "required_event_field_invalid",
                    "events",
                    f"{field} must be an integer >= {minimum}",
                    match_id=event.match_id,
                    source_event_id=event.source_event_id,
                )
        if not isinstance(event.timestamp, time):
            self._issue(
                issues,
                "event_timestamp_invalid",
                "events",
                "timestamp must be a period-local time",
                match_id=event.match_id,
                source_event_id=event.source_event_id,
            )
        for field in ("under_pressure", "counterpress"):
            if not isinstance(getattr(event, field), bool):
                self._issue(
                    issues,
                    "event_boolean_invalid",
                    "events",
                    f"{field} must be boolean",
                    match_id=event.match_id,
                    source_event_id=event.source_event_id,
                )
        if not isinstance(event.related_event_ids, (tuple, list)) or any(
            not isinstance(value, UUID) for value in (event.related_event_ids or ())
        ):
            self._issue(
                issues,
                "related_event_ids_invalid",
                "events",
                "related_event_ids must contain only UUID values",
                match_id=event.match_id,
                source_event_id=event.source_event_id,
            )

        for field in ("event_type", "event_subtype", "body_part", "play_pattern"):
            value = getattr(event, field)
            if value is not None and (
                not isinstance(value, str) or not SNAKE_CASE_PATTERN.fullmatch(value)
            ):
                self._issue(
                    issues,
                    "canonical_text_invalid",
                    "events",
                    f"{field} must be lowercase snake_case when present",
                    match_id=event.match_id,
                    source_event_id=event.source_event_id,
                )

        if not isinstance(event.raw_details, dict):
            self._issue(
                issues,
                "raw_details_invalid",
                "events",
                "raw_details must be a JSON object",
                match_id=event.match_id,
                source_event_id=event.source_event_id,
            )
        else:
            self._validate_subtype_preservation(event, issues)

    def _validate_subtype_preservation(
        self, event: CanonicalEvent, issues: list[ContractIssue]
    ) -> None:
        detail_key = EVENT_DETAIL_KEYS.get(event.event_type)
        detail = event.raw_details.get(detail_key) if detail_key else None
        if not isinstance(detail, dict) or detail.get("type") is None:
            return
        source_type = detail.get("type")
        source_name = source_type.get("name") if isinstance(source_type, dict) else None
        if not isinstance(source_name, str):
            self._issue(
                issues,
                "source_subtype_invalid",
                "events",
                f"raw_details.{detail_key}.type.name must be text",
                match_id=event.match_id,
                source_event_id=event.source_event_id,
            )
            return
        expected = re.sub(r"[^a-z0-9]+", "_", source_name.casefold()).strip("_")
        if event.event_subtype != expected:
            self._issue(
                issues,
                "event_subtype_not_preserved",
                "events",
                f"event_subtype {event.event_subtype!r} does not match "
                f"raw source subtype {expected!r}",
                match_id=event.match_id,
                source_event_id=event.source_event_id,
            )

    def _validate_possessions(
        self,
        events: Sequence[CanonicalEvent],
        issues: list[ContractIssue],
    ) -> None:
        possession_teams: dict[tuple[int, int], set[int]] = defaultdict(set)
        previous_possession: dict[int, int] = {}
        for event in events:
            if event.event_type in POSSESSION_ORDER_EXEMPT_TYPES:
                continue
            if not isinstance(event.possession_id, int):
                continue
            if isinstance(event.possession_team_id, int):
                possession_teams[(event.match_id, event.possession_id)].add(
                    event.possession_team_id
                )
            prior = previous_possession.get(event.match_id)
            if prior is not None and event.possession_id < prior:
                self._issue(
                    issues,
                    "possession_order_invalid",
                    "possessions",
                    "possession_id decreased in authoritative event order",
                    match_id=event.match_id,
                    source_event_id=event.source_event_id,
                )
            previous_possession[event.match_id] = event.possession_id

        for (match_id, possession_id), teams in possession_teams.items():
            if len(teams) > 1:
                self._issue(
                    issues,
                    "possession_team_inconsistent",
                    "possessions",
                    f"possession {possession_id} has multiple possession teams: "
                    f"{sorted(teams)}",
                    match_id=match_id,
                )

    def _validate_lineups(
        self,
        events: Sequence[CanonicalEvent],
        intervals: Sequence[CanonicalLineupInterval],
        issues: list[ContractIssue],
        warnings: list[ContractIssue],
    ) -> None:
        match_ids = {event.match_id for event in events}
        known_players: dict[int, set[int]] = defaultdict(set)
        seen: set[tuple[str, int, int, int, int]] = set()
        for interval in intervals:
            key = (
                interval.source,
                interval.match_id,
                interval.player_id,
                interval.position_id,
                interval.from_seconds,
            )
            if key in seen:
                self._issue(
                    issues,
                    "duplicate_lineup_interval",
                    "player_match_intervals",
                    "lineup interval identity must be unique",
                    match_id=interval.match_id,
                )
            seen.add(key)
            if interval.match_id not in match_ids:
                self._issue(
                    issues,
                    "lineup_match_not_found",
                    "player_match_intervals",
                    "lineup interval does not belong to an input match",
                    match_id=interval.match_id,
                )
            if (
                isinstance(interval.to_seconds, int)
                and isinstance(interval.from_seconds, int)
                and interval.to_seconds < interval.from_seconds
            ):
                self._issue(
                    warnings,
                    "lineup_interval_reversed",
                    "player_match_intervals",
                    "source tactical interval ends before it starts; retained as "
                    "a non-blocking quality warning",
                    match_id=interval.match_id,
                )
            known_players[interval.match_id].add(interval.player_id)

        event_people = {
            (event.match_id, person_id, role)
            for event in events
            for person_id, role in (
                (event.player_id, "player"),
                (event.recipient_id, "recipient"),
            )
            if person_id is not None
            and event.event_type in SOCCERACTION_ACTIONABLE_EVENT_TYPES
        }
        for match_id, player_id, role in sorted(event_people):
            if player_id not in known_players.get(match_id, set()):
                self._issue(
                    issues,
                    f"event_{role}_missing_lineup_interval",
                    "player_match_intervals",
                    f"event {role} {player_id} has no lineup interval",
                    match_id=match_id,
                )

    def _validate_three_sixty(
        self,
        events: Sequence[CanonicalEvent],
        frames: Sequence[Canonical360Frame],
        issues: list[ContractIssue],
    ) -> None:
        event_match = {
            (event.source, event.source_event_id): event.match_id for event in events
        }
        seen: set[tuple[str, UUID]] = set()
        for frame in frames:
            key = (frame.source, frame.source_event_id)
            if key in seen:
                self._issue(
                    issues,
                    "duplicate_three_sixty",
                    "event_360",
                    "only one 360 frame is allowed per source event",
                    match_id=frame.match_id,
                    source_event_id=frame.source_event_id,
                )
            seen.add(key)
            linked_match = event_match.get(key)
            if linked_match is None:
                self._issue(
                    issues,
                    "three_sixty_event_not_found",
                    "event_360",
                    "360 frame does not link to an input event",
                    match_id=frame.match_id,
                    source_event_id=frame.source_event_id,
                )
            elif linked_match != frame.match_id:
                self._issue(
                    issues,
                    "three_sixty_match_mismatch",
                    "event_360",
                    f"360 frame match {frame.match_id} differs from event match "
                    f"{linked_match}",
                    match_id=frame.match_id,
                    source_event_id=frame.source_event_id,
                )
