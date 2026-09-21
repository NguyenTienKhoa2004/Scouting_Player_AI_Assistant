"""Shared primitive checks and issue collection for raw validation."""

from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from ..reader import RawRecord
from .models import RawValidationIssue


_CLOCK_PATTERN = re.compile(r"^(\d+):(\d{2})$")


class RawValidationContext:
    def __init__(self, max_issues: int) -> None:
        self.max_issues = max_issues
        self.issues: list[RawValidationIssue] = []
        self.blocking_issues: list[RawValidationIssue] = []
        self.deferred_issue_count = 0
        self.truncated = False

    def add(self, record: RawRecord, code: str, field: str, message: str) -> None:
        self._append(
            RawValidationIssue(
                code=code,
                kind=record.kind,
                match_id=record.match_id,
                source_file=str(record.source_file),
                source_record_index=record.source_record_index,
                field=field,
                message=message,
            )
        )

    def add_global(self, code: str, message: str, source: Path) -> None:
        self._append(
            RawValidationIssue(
                code=code,
                kind="dataset",
                match_id=None,
                source_file=str(source),
                source_record_index=None,
                field="source",
                message=message,
            )
        )

    def _append(self, issue: RawValidationIssue) -> None:
        if issue.is_blocking:
            if len(self.blocking_issues) < self.max_issues:
                self.blocking_issues.append(issue)
        else:
            self.deferred_issue_count += 1
        if len(self.issues) >= self.max_issues:
            self.truncated = True
            return
        self.issues.append(issue)

    def object(
        self, record: RawRecord, value: Any, field: str
    ) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            self.add(record, "object_required", field, f"{field} must be an object")
            return None
        return value

    def positive_int(
        self, record: RawRecord, value: Any, field: str
    ) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            self.add(
                record,
                "positive_integer_required",
                field,
                f"{field} must be a positive integer",
            )
            return None
        return value

    def nonnegative_int(
        self, record: RawRecord, value: Any, field: str
    ) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self.add(
                record,
                "nonnegative_integer_required",
                field,
                f"{field} must be a nonnegative integer",
            )
            return None
        return value

    def bounded_int(
        self,
        record: RawRecord,
        value: Any,
        field: str,
        minimum: int,
        maximum: int,
    ) -> int | None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            self.add(
                record,
                "integer_out_of_range",
                field,
                f"{field} must be an integer in {minimum}..{maximum}",
            )
            return None
        return value

    def period(
        self,
        record: RawRecord,
        value: Any,
        field: str,
        *,
        required: bool,
    ) -> int | None:
        if value is None and not required:
            return None
        return self.bounded_int(record, value, field, 1, 5)

    def nonempty_text(
        self, record: RawRecord, value: Any, field: str
    ) -> str | None:
        if not isinstance(value, str) or not value.strip():
            self.add(
                record,
                "nonempty_text_required",
                field,
                f"{field} must be non-empty text",
            )
            return None
        return value

    def uuid(self, record: RawRecord, value: Any, field: str) -> UUID | None:
        if not isinstance(value, str):
            self.add(record, "uuid_required", field, f"{field} must be UUID text")
            return None
        try:
            result = UUID(value)
        except ValueError:
            self.add(record, "uuid_invalid", field, f"{field} is not a valid UUID")
            return None
        if result.int == 0:
            self.add(record, "uuid_nil", field, f"{field} must not be the nil UUID")
            return None
        return result

    def finite_number(
        self, record: RawRecord, value: Any, field: str
    ) -> float | None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            self.add(
                record,
                "finite_number_required",
                field,
                f"{field} must be a finite number",
            )
            return None
        return float(value)

    def bounded_number(
        self,
        record: RawRecord,
        value: Any,
        field: str,
        minimum: float,
        maximum: float,
    ) -> float | None:
        number = self.finite_number(record, value, field)
        if number is not None and not minimum <= number <= maximum:
            self.add(
                record,
                "number_out_of_range",
                field,
                f"{field} must be in {minimum:g}..{maximum:g}",
            )
            return None
        return number

    def optional_nonnegative_number(
        self, record: RawRecord, value: Any, field: str
    ) -> float | None:
        if value is None:
            return None
        number = self.finite_number(record, value, field)
        if number is not None and number < 0:
            self.add(
                record,
                "number_negative",
                field,
                f"{field} must be nonnegative",
            )
            return None
        return number

    def optional_bool(self, record: RawRecord, value: Any, field: str) -> None:
        if value is not None and not isinstance(value, bool):
            self.add(
                record,
                "boolean_required",
                field,
                f"{field} must be boolean when present",
            )

    def clock(
        self,
        record: RawRecord,
        value: Any,
        field: str,
        *,
        required: bool,
    ) -> None:
        if value is None and not required:
            return
        if not isinstance(value, str):
            self.add(
                record,
                "lineup_clock_invalid",
                field,
                f"{field} must be MM:SS text",
            )
            return
        match = _CLOCK_PATTERN.fullmatch(value)
        if match is None or int(match.group(2)) > 59:
            self.add(
                record,
                "lineup_clock_invalid",
                field,
                f"{field} must be MM:SS with seconds in 00..59",
            )

    def iso_date(self, record: RawRecord, value: Any, field: str) -> None:
        if not isinstance(value, str):
            self.add(record, "date_invalid", field, f"{field} must be an ISO date")
            return
        try:
            date.fromisoformat(value)
        except ValueError:
            self.add(record, "date_invalid", field, f"{field} must be an ISO date")

    def location(
        self,
        record: RawRecord,
        value: Any,
        field: str,
        *,
        required: bool,
    ) -> None:
        if value is None:
            if required:
                self.add(record, "location_required", field, f"{field} is required")
            return
        if not isinstance(value, list) or len(value) < 2:
            self.add(
                record,
                "location_invalid",
                field,
                f"{field} must contain x and y",
            )
            return
        self.bounded_number(record, value[0], f"{field}[0]", 0.0, 120.0)
        self.bounded_number(record, value[1], f"{field}[1]", 0.0, 80.0)


__all__ = ["RawValidationContext"]
