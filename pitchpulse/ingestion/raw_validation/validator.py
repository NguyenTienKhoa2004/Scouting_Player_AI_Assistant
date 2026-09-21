"""Orchestrate validation of immutable StatsBomb source data."""

from __future__ import annotations

from uuid import UUID

from ..reader import RawDataError, StatsBombRawReader
from .context import RawValidationContext
from .event_validator import validate_events
from .lineup_validator import validate_lineups
from .match_validator import validate_match
from .models import RawStatsBombValidationReport
from .three_sixty_validator import validate_three_sixty


class RawStatsBombValidator:
    """Validate a complete source selection before normalization or persistence."""

    def __init__(self, *, max_issues: int = 1000) -> None:
        if max_issues <= 0:
            raise ValueError("max_issues must be positive")
        self.max_issues = max_issues

    def validate(self, reader: StatsBombRawReader) -> RawStatsBombValidationReport:
        context = RawValidationContext(self.max_issues)
        match_count = 0
        lineup_count = 0
        event_count = 0
        three_sixty_count = 0
        seen_match_ids: set[int] = set()
        seen_event_ids: dict[UUID, int] = {}

        try:
            for bundle in reader.iter_match_bundles():
                match_count += 1
                lineup_count += len(bundle.lineups)
                event_count += len(bundle.events)
                three_sixty_count += len(bundle.three_sixty)

                if bundle.match.match_id in seen_match_ids:
                    context.add(
                        bundle.match,
                        "duplicate_match_id",
                        "match_id",
                        f"match {bundle.match.match_id} occurs more than once",
                    )
                seen_match_ids.add(bundle.match.match_id)

                team_ids = validate_match(bundle.match, reader, context)
                player_teams = validate_lineups(bundle, team_ids, context)
                event_ids = validate_events(
                    bundle,
                    team_ids,
                    player_teams,
                    seen_event_ids,
                    context,
                )
                validate_three_sixty(bundle, event_ids, context)
        except RawDataError as exc:
            context.add_global("raw_data_unreadable", str(exc), reader.data_root)

        return RawStatsBombValidationReport(
            match_count=match_count,
            lineup_record_count=lineup_count,
            event_count=event_count,
            three_sixty_count=three_sixty_count,
            issues=tuple(context.issues),
            blocking_issues=tuple(context.blocking_issues),
            deferred_issue_count=context.deferred_issue_count,
            truncated=context.truncated,
        )

__all__ = ["RawStatsBombValidator"]
