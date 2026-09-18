"""Validation rules for normalized StatsBomb 360 frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any
from uuid import UUID

from .three_sixty_normalizer import Canonical360Frame
from .validator import ValidationIssue


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
        if (
            known_event_ids is not None
            and frame.source_event_id not in known_event_ids
        ):
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
            self._validate_flat_coordinates(frame.visible_area, "visible_area", issues)

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


__all__ = ["Canonical360Validator", "ThreeSixtyValidationResult"]
