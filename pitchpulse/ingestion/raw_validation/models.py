"""Result types for raw StatsBomb validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# These source-level issues have an equivalent normalizer/canonical validation
# rule. They must remain visible in the raw report, but the affected record can
# safely continue to the canonical stage where it is written to quarantine.
DEFERRED_CANONICAL_ISSUE_CODES = frozenset(
    {
        "freeze_frame_multiple_actors",
        "location_required",
        "number_negative",
        "number_out_of_range",
        "outcome_required",
        "player_required",
        "time_fields_inconsistent",
        "unknown_event_type",
    }
)


@dataclass(frozen=True, slots=True)
class RawValidationIssue:
    code: str
    kind: str
    match_id: int | None
    source_file: str
    source_record_index: int | None
    field: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_blocking(self) -> bool:
        """Whether the issue makes selection-level ingestion unsafe."""
        return self.code not in DEFERRED_CANONICAL_ISSUE_CODES


@dataclass(frozen=True, slots=True)
class RawStatsBombValidationReport:
    match_count: int
    lineup_record_count: int
    event_count: int
    three_sixty_count: int
    issues: tuple[RawValidationIssue, ...]
    blocking_issues: tuple[RawValidationIssue, ...] = ()
    deferred_issue_count: int = 0
    truncated: bool = False

    @property
    def is_valid(self) -> bool:
        return not self.issues and not self.truncated

    @property
    def is_ingestible(self) -> bool:
        """Whether preflight found no selection-level integrity blockers."""
        return not self.blocking_issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "match_count": self.match_count,
            "lineup_record_count": self.lineup_record_count,
            "event_count": self.event_count,
            "three_sixty_count": self.three_sixty_count,
            "issues": [issue.as_dict() for issue in self.issues],
            "blocking_issues": [
                issue.as_dict() for issue in self.blocking_issues
            ],
            "deferred_issue_count": self.deferred_issue_count,
            "truncated": self.truncated,
        }


class RawStatsBombValidationError(ValueError):
    def __init__(self, report: RawStatsBombValidationReport) -> None:
        self.report = report
        preview_issues = report.blocking_issues or report.issues
        preview = "; ".join(
            f"{issue.code} at match={issue.match_id} "
            f"record={issue.source_record_index} field={issue.field}: "
            f"{issue.message}"
            for issue in preview_issues[:5]
        )
        remaining = len(preview_issues) - 5
        if remaining > 0:
            preview += f"; and {remaining} more blocking issues"
        if report.truncated:
            preview += "; additional non-blocking issue details were truncated"
        super().__init__(f"raw StatsBomb validation failed: {preview}")


__all__ = [
    "DEFERRED_CANONICAL_ISSUE_CODES",
    "RawStatsBombValidationError",
    "RawStatsBombValidationReport",
    "RawValidationIssue",
]
