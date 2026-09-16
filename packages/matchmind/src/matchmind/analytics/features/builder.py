"""Build an in-memory analytics dataset from validated SPADL input."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .vaep_features import (
    BASE_FEATURE_VERSION,
    ActionFeatureBuilder,
    ActionFeatureRow,
)
from .action_state import ActionState, ActionStateBuilder, STATE_CONTRACT_VERSION
from .features_360 import FEATURE_360_VERSION, ThreeSixtyFeatureEnricher
from .spadl_converter import (
    EventToActionConverter,
    SpadlAction,
    SpadlConversionReport,
)
from matchmind.validator.spadl_input_validator import SpadlInput


@dataclass(frozen=True, slots=True)
class AnalyticsDataset:
    actions: tuple[SpadlAction, ...]
    states: tuple[ActionState, ...]
    features: tuple[ActionFeatureRow, ...]
    conversion_report: SpadlConversionReport
    feature_version: str
    include_360: bool
    input_warning_count: int
    input_warnings_by_code: tuple[tuple[str, int], ...]

    def quality_report(self) -> dict[str, Any]:
        actions_with_360 = sum(action.has_360 for action in self.actions)
        return {
            **self.conversion_report.as_dict(),
            "state_contract_version": STATE_CONTRACT_VERSION,
            "feature_version": self.feature_version,
            "include_360": self.include_360,
            "input_warning_count": self.input_warning_count,
            "input_warnings_by_code": dict(self.input_warnings_by_code),
            "state_count": len(self.states),
            "feature_count": len(self.features),
            "actions_with_360": actions_with_360,
            "actions_without_360": len(self.actions) - actions_with_360,
            "possession_transition_count": sum(
                state.possession_changed for state in self.states
            ),
            "missing_player_count": sum(
                action.player_id is None for action in self.actions
            ),
            "coordinate_violation_count": sum(
                not all(
                    0.0 <= coordinate <= limit
                    for coordinate, limit in (
                        (action.start_x, 105.0),
                        (action.end_x, 105.0),
                        (action.start_y, 68.0),
                        (action.end_y, 68.0),
                    )
                )
                for action in self.actions
            ),
        }


class AnalyticsDatasetBuilder:
    def build(self, inputs: SpadlInput, *, include_360: bool = False) -> AnalyticsDataset:
        conversion = EventToActionConverter().convert(inputs)
        states = ActionStateBuilder().build(conversion.actions, inputs)
        features = ActionFeatureBuilder().build(states, inputs)
        feature_version = BASE_FEATURE_VERSION
        if include_360:
            features = ThreeSixtyFeatureEnricher().enrich(
                features, conversion.actions, inputs
            )
            feature_version = FEATURE_360_VERSION
        return AnalyticsDataset(
            actions=conversion.actions,
            states=states,
            features=features,
            conversion_report=conversion.report,
            feature_version=feature_version,
            include_360=include_360,
            input_warning_count=len(inputs.report.warnings),
            input_warnings_by_code=tuple(
                sorted(
                    Counter(
                        warning.code for warning in inputs.report.warnings
                    ).items()
                )
            ),
        )


__all__ = ["AnalyticsDataset", "AnalyticsDatasetBuilder"]
