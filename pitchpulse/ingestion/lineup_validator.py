"""Validation rules for normalized lineup intervals."""

from __future__ import annotations

from dataclasses import dataclass

from .lineup_normalizer import CanonicalLineupInterval
from .validator import PERIOD_START_MINUTES, ValidationIssue


@dataclass(frozen=True, slots=True)
class LineupIntervalValidationResult:
    interval: CanonicalLineupInterval
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


class CanonicalLineupIntervalValidator:
    """Validate one playing interval."""

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


__all__ = ["CanonicalLineupIntervalValidator", "LineupIntervalValidationResult"]
