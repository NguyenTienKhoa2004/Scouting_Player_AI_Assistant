"""Leakage-safe score and possession state after every SPADL action."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .spadl_converter import SpadlAction
from matchmind.validator.spadl_input_validator import SpadlInput


STATE_CONTRACT_VERSION = "spadl-action-state-v1"
GOAL_ACTION_TYPES = frozenset({"shot", "shot_penalty", "shot_freekick"})


class StateConstructionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ActionState:
    """Match state immediately after the referenced action."""

    action: SpadlAction
    state_version: str
    home_score: int
    away_score: int
    score_for: int
    score_against: int
    goal_difference: int
    possession_changed: bool
    acting_team_has_possession: bool
    is_shootout: bool

    def as_dict(self) -> dict[str, Any]:
        values = self.action.as_dict()
        values.update(
            {
                "state_version": self.state_version,
                "home_score": self.home_score,
                "away_score": self.away_score,
                "score_for": self.score_for,
                "score_against": self.score_against,
                "goal_difference": self.goal_difference,
                "possession_changed": self.possession_changed,
                "acting_team_has_possession": self.acting_team_has_possession,
                "is_shootout": self.is_shootout,
            }
        )
        return values


class ActionStateBuilder:
    """Build score and possession state in action order without future data."""

    def build(
        self, actions: tuple[SpadlAction, ...], inputs: SpadlInput
    ) -> tuple[ActionState, ...]:
        scores: defaultdict[int, dict[int, int]] = defaultdict(dict)
        previous_possession: dict[int, tuple[int, int]] = {}
        expected_action_id: defaultdict[int, int] = defaultdict(int)
        states: list[ActionState] = []

        for action in actions:
            if action.action_id != expected_action_id[action.match_id]:
                raise StateConstructionError(
                    f"match {action.match_id} expected action_id "
                    f"{expected_action_id[action.match_id]}, got {action.action_id}"
                )
            expected_action_id[action.match_id] += 1

            home_team = inputs.home_team_by_match.get(action.match_id)
            away_team = inputs.away_team_by_match.get(action.match_id)
            if home_team is None or away_team is None:
                raise StateConstructionError(
                    f"match {action.match_id} has incomplete team context"
                )
            match_scores = scores[action.match_id]
            match_scores.setdefault(home_team, 0)
            match_scores.setdefault(away_team, 0)
            if action.team_id not in match_scores:
                raise StateConstructionError(
                    f"action team {action.team_id} does not belong to match "
                    f"{action.match_id}"
                )

            is_shootout = action.period_id == 5
            if not is_shootout:
                if (
                    action.type_name in GOAL_ACTION_TYPES
                    and action.result_name == "success"
                ):
                    match_scores[action.team_id] += 1
                elif action.result_name == "owngoal":
                    opponent = away_team if action.team_id == home_team else home_team
                    match_scores[opponent] += 1

            possession_key = (action.possession_id, action.possession_team_id)
            prior = previous_possession.get(action.match_id)
            possession_changed = prior is not None and prior != possession_key
            previous_possession[action.match_id] = possession_key

            score_for = match_scores[action.team_id]
            opponent = away_team if action.team_id == home_team else home_team
            score_against = match_scores[opponent]
            states.append(
                ActionState(
                    action=action,
                    state_version=STATE_CONTRACT_VERSION,
                    home_score=match_scores[home_team],
                    away_score=match_scores[away_team],
                    score_for=score_for,
                    score_against=score_against,
                    goal_difference=score_for - score_against,
                    possession_changed=possession_changed,
                    acting_team_has_possession=(
                        action.team_id == action.possession_team_id
                    ),
                    is_shootout=is_shootout,
                )
            )

        return tuple(states)


def build_action_states(
    actions: tuple[SpadlAction, ...], inputs: SpadlInput
) -> tuple[ActionState, ...]:
    return ActionStateBuilder().build(actions, inputs)


__all__ = [
    "STATE_CONTRACT_VERSION",
    "ActionState",
    "ActionStateBuilder",
    "StateConstructionError",
    "build_action_states",
]
