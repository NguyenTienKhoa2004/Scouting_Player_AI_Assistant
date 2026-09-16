"""Input contract for StatsBomb-to-SPADL conversion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from matchmind.ingestion.normalizer import (
    Canonical360Frame,
    CanonicalEvent,
    CanonicalLineupInterval,
)


ACTIONABLE_EVENT_TYPES = frozenset(
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


@dataclass(frozen=True, slots=True)
class ContractIssue:
    code: str
    entity: str
    message: str
    match_id: int | None = None
    source_event_id: UUID | None = None

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["source_event_id"] = (
            str(self.source_event_id) if self.source_event_id else None
        )
        return values


@dataclass(frozen=True, slots=True)
class SpadlInputValidationReport:
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
            "issues": [issue.as_dict() for issue in self.issues],
            "warnings": [warning.as_dict() for warning in self.warnings],
        }


class SpadlInputContractError(RuntimeError):
    def __init__(self, report: SpadlInputValidationReport) -> None:
        self.report = report
        preview = "; ".join(
            f"{issue.code}: {issue.message}" for issue in report.issues[:3]
        )
        if len(report.issues) > 3:
            preview += f"; and {len(report.issues) - 3} more"
        super().__init__(f"SPADL input contract failed: {preview}")


@dataclass(frozen=True, slots=True)
class SpadlInput:
    events: tuple[CanonicalEvent, ...]
    lineup_intervals: tuple[CanonicalLineupInterval, ...]
    three_sixty_by_event: dict[tuple[str, UUID], Canonical360Frame]
    home_team_by_match: dict[int, int]
    away_team_by_match: dict[int, int]
    report: SpadlInputValidationReport


def validate_spadl_input(
    events: Sequence[CanonicalEvent],
    intervals: Sequence[CanonicalLineupInterval],
    frames: Sequence[Canonical360Frame],
) -> SpadlInputValidationReport:
    """Check only cross-record invariants not guaranteed by row validation."""
    issues: list[ContractIssue] = []
    warnings: list[ContractIssue] = []

    if not events:
        issues.append(ContractIssue("events_empty", "events", "no events found"))

    possession_teams: dict[tuple[int, int], set[int]] = defaultdict(set)
    for event in events:
        possession_teams[event.match_id, event.possession_id].add(
            event.possession_team_id
        )
    for (match_id, possession_id), teams in possession_teams.items():
        if len(teams) > 1:
            issues.append(
                ContractIssue(
                    "possession_team_inconsistent",
                    "possessions",
                    f"possession {possession_id} has teams {sorted(teams)}",
                    match_id,
                )
            )

    previous: dict[int, int] = {}
    actionable = (
        event
        for event in events
        if event.event_type in ACTIONABLE_EVENT_TYPES
    )
    for event in sorted(
        actionable,
        key=lambda item: (
            item.match_id,
            item.period,
            item.timestamp,
            item.source_event_index,
        ),
    ):
        if (
            event.match_id in previous
            and event.possession_id < previous[event.match_id]
        ):
            issues.append(
                ContractIssue(
                    "possession_order_invalid",
                    "possessions",
                    "possession_id decreased in action order",
                    event.match_id,
                    event.source_event_id,
                )
            )
        previous[event.match_id] = event.possession_id

    players_by_match: dict[int, set[int]] = defaultdict(set)
    event_match_ids = {event.match_id for event in events}
    for interval in intervals:
        players_by_match[interval.match_id].add(interval.player_id)
        if interval.match_id not in event_match_ids:
            issues.append(
                ContractIssue(
                    "lineup_match_not_found",
                    "player_match_intervals",
                    "lineup interval has no input match",
                    interval.match_id,
                )
            )
        if (
            interval.to_seconds is not None
            and interval.to_seconds < interval.from_seconds
        ):
            warnings.append(
                ContractIssue(
                    "lineup_interval_reversed",
                    "player_match_intervals",
                    "lineup interval ends before it starts",
                    interval.match_id,
                )
            )

    missing_people = {
        (event.match_id, person_id, role)
        for event in events
        if event.event_type in ACTIONABLE_EVENT_TYPES
        for person_id, role in (
            (event.player_id, "player"),
            (event.recipient_id, "recipient"),
        )
        if person_id is not None
        and person_id not in players_by_match[event.match_id]
    }
    for match_id, person_id, role in sorted(missing_people):
        issues.append(
            ContractIssue(
                f"event_{role}_missing_lineup_interval",
                "player_match_intervals",
                f"event {role} {person_id} has no lineup interval",
                match_id,
            )
        )

    event_matches = {
        (event.source, event.source_event_id): event.match_id for event in events
    }
    for frame in frames:
        linked_match = event_matches.get((frame.source, frame.source_event_id))
        if linked_match is None:
            issues.append(
                ContractIssue(
                    "three_sixty_event_not_found",
                    "event_360",
                    "360 frame has no input event",
                    frame.match_id,
                    frame.source_event_id,
                )
            )
        elif linked_match != frame.match_id:
            issues.append(
                ContractIssue(
                    "three_sixty_match_mismatch",
                    "event_360",
                    f"360 frame links to match {linked_match}",
                    frame.match_id,
                    frame.source_event_id,
                )
            )

    return SpadlInputValidationReport(
        match_count=len(event_match_ids),
        event_count=len(events),
        lineup_interval_count=len(intervals),
        three_sixty_count=len(frames),
        events_with_subtype=sum(event.event_subtype is not None for event in events),
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


__all__ = [
    "ContractIssue",
    "SpadlInput",
    "SpadlInputContractError",
    "SpadlInputValidationReport",
    "validate_spadl_input",
]
