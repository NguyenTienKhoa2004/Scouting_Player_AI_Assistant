"""Standard socceraction VAEP feature construction."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd
from socceraction.vaep import VAEP

from .action_state import ActionState
from matchmind.spadl.converter import SOCCERACTION_VERSION
from matchmind.spadl.input_validator import SpadlInput


NB_PREV_ACTIONS = 3
BASE_FEATURE_VERSION = (
    f"socceraction-{SOCCERACTION_VERSION}-vaep-features-{NB_PREV_ACTIONS}actions-v1"
)
FeatureValue = bool | int | float | str | None


@dataclass(frozen=True, slots=True)
class ActionFeatureRow:
    """One standard VAEP feature vector keyed by match and action."""

    match_id: int
    action_id: int
    feature_version: str
    values: tuple[tuple[str, FeatureValue], ...]

    def as_dict(self) -> dict[str, FeatureValue]:
        return {
            "match_id": self.match_id,
            "action_id": self.action_id,
            "feature_version": self.feature_version,
            **dict(self.values),
        }


class ActionFeatureBuilder:
    """Use socceraction's default VAEP transformers with a three-action state."""

    def __init__(self) -> None:
        self.vaep = VAEP(nb_prev_actions=NB_PREV_ACTIONS)

    def build(
        self,
        states: tuple[ActionState, ...],
        inputs: SpadlInput,
    ) -> tuple[ActionFeatureRow, ...]:
        states_by_match: defaultdict[int, list[ActionState]] = defaultdict(list)
        for state in states:
            states_by_match[state.action.match_id].append(state)

        rows: list[ActionFeatureRow] = []
        for match_id in sorted(states_by_match):
            match_states = states_by_match[match_id]
            actions = pd.DataFrame(
                [self._socceraction_row(state) for state in match_states]
            )
            game = pd.Series(
                {
                    "game_id": match_id,
                    "home_team_id": inputs.home_team_by_match[match_id],
                }
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning, module=r"socceraction\..*"
                )
                features = self.vaep.compute_features(game, actions)
            if len(features) != len(match_states):
                raise ValueError("socceraction returned a mismatched feature count")

            for position, (_, feature_row) in enumerate(features.iterrows()):
                action = match_states[position].action
                values = tuple(
                    (str(name), self._python_value(value))
                    for name, value in feature_row.items()
                )
                rows.append(
                    ActionFeatureRow(
                        match_id=match_id,
                        action_id=action.action_id,
                        feature_version=BASE_FEATURE_VERSION,
                        values=values,
                    )
                )
        return tuple(rows)

    @staticmethod
    def _socceraction_row(state: ActionState) -> dict[str, Any]:
        action = state.action
        return {
            "game_id": action.match_id,
            "original_event_id": str(action.source_event_id),
            "action_id": action.action_id,
            "period_id": action.period_id,
            "time_seconds": action.time_seconds,
            "team_id": action.team_id,
            "player_id": action.player_id,
            "start_x": action.start_x,
            "start_y": action.start_y,
            "end_x": action.end_x,
            "end_y": action.end_y,
            "type_id": action.type_id,
            "result_id": action.result_id,
            "bodypart_id": action.bodypart_id,
        }

    @staticmethod
    def _python_value(value: Any) -> FeatureValue:
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        return str(value)


def build_action_features(
    states: tuple[ActionState, ...], inputs: SpadlInput
) -> tuple[ActionFeatureRow, ...]:
    return ActionFeatureBuilder().build(states, inputs)


__all__ = [
    "BASE_FEATURE_VERSION",
    "NB_PREV_ACTIONS",
    "ActionFeatureBuilder",
    "ActionFeatureRow",
    "FeatureValue",
    "build_action_features",
]
