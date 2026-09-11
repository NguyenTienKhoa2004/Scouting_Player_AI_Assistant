"""Optional StatsBomb 360 features behind an explicit contract version."""

from __future__ import annotations

import math

from .vaep_features import BASE_FEATURE_VERSION, ActionFeatureRow, FeatureValue
from .spadl_converter import SPADL_FIELD_LENGTH, SPADL_FIELD_WIDTH, SpadlAction
from .input import SpadlInput


FEATURE_360_VERSION = f"{BASE_FEATURE_VERSION}-360-v1"
PRESSURE_RADIUS_METRES = 5.0
PASSING_LANE_HALF_WIDTH_METRES = 2.0
GOAL_HALF_WIDTH_METRES = 3.66


class ThreeSixtyFeatureEnricher:
    """Add spatial context without changing the baseline feature contract."""

    def enrich(
        self,
        rows: tuple[ActionFeatureRow, ...],
        actions: tuple[SpadlAction, ...],
        inputs: SpadlInput,
    ) -> tuple[ActionFeatureRow, ...]:
        actions_by_key = {
            (action.match_id, action.action_id): action for action in actions
        }
        enriched: list[ActionFeatureRow] = []
        for row in rows:
            action = actions_by_key[(row.match_id, row.action_id)]
            spatial = self._spatial_features(action, inputs)
            enriched.append(
                ActionFeatureRow(
                    match_id=row.match_id,
                    action_id=row.action_id,
                    feature_version=FEATURE_360_VERSION,
                    values=row.values + tuple(spatial.items()),
                )
            )
        return tuple(enriched)

    def _spatial_features(
        self, action: SpadlAction, inputs: SpadlInput
    ) -> dict[str, FeatureValue]:
        if action.synthetic:
            return self._missing_features()
        frame = inputs.three_sixty_by_event.get(
            (action.source, action.source_event_id)
        )
        if frame is None:
            return self._missing_features()

        home_team_id = inputs.home_team_by_match[action.match_id]
        teammates: list[tuple[float, float]] = []
        opponents: list[tuple[float, float]] = []
        for player in frame.freeze_frame:
            location = player.get("location")
            if (
                not isinstance(location, list)
                or len(location) < 2
                or player.get("actor") is True
            ):
                continue
            point = self._frame_point(
                float(location[0]),
                float(location[1]),
                action.team_id,
                home_team_id,
            )
            if player.get("teammate") is True:
                teammates.append(point)
            else:
                opponents.append(point)

        opponent_distances = [
            math.hypot(x - action.start_x, y - action.start_y)
            for x, y in opponents
        ]
        visible_polygon = [
            self._frame_point(x, y, action.team_id, home_team_id)
            for x, y in zip(frame.visible_area[::2], frame.visible_area[1::2])
        ]
        attacking_goal_x = (
            SPADL_FIELD_LENGTH if action.team_id == home_team_id else 0.0
        )
        goal_center = (attacking_goal_x, SPADL_FIELD_WIDTH / 2.0)
        goal_visible = self._point_in_polygon(goal_center, visible_polygon)
        return {
            "has_360": True,
            "visible_teammates": len(teammates),
            "visible_opponents": len(opponents),
            "nearest_defender_distance": (
                min(opponent_distances) if opponent_distances else None
            ),
            "pressure_around_ball": sum(
                distance <= PRESSURE_RADIUS_METRES
                for distance in opponent_distances
            ),
            "passing_lane_obstruction": sum(
                self._distance_to_segment(
                    opponent,
                    (action.start_x, action.start_y),
                    (action.end_x, action.end_y),
                )
                <= PASSING_LANE_HALF_WIDTH_METRES
                for opponent in opponents
            ),
            "goal_visible": goal_visible,
            "visible_goal_angle": (
                self._goal_angle(
                    action.start_x, action.start_y, attacking_goal_x
                )
                if goal_visible
                else None
            ),
        }

    @staticmethod
    def _missing_features() -> dict[str, FeatureValue]:
        return {
            "has_360": False,
            "visible_teammates": None,
            "visible_opponents": None,
            "nearest_defender_distance": None,
            "pressure_around_ball": None,
            "passing_lane_obstruction": None,
            "goal_visible": None,
            "visible_goal_angle": None,
        }

    @staticmethod
    def _frame_point(
        x: float, y: float, team_id: int, home_team_id: int
    ) -> tuple[float, float]:
        point_x = x / 120.0 * SPADL_FIELD_LENGTH
        point_y = (80.0 - y) / 80.0 * SPADL_FIELD_WIDTH
        if team_id != home_team_id:
            point_x = SPADL_FIELD_LENGTH - point_x
            point_y = SPADL_FIELD_WIDTH - point_y
        return point_x, point_y

    @staticmethod
    def _distance_to_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        projection = (
            (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
        ) / length_squared
        projection = min(1.0, max(0.0, projection))
        nearest = (start[0] + projection * dx, start[1] + projection * dy)
        return math.hypot(point[0] - nearest[0], point[1] - nearest[1])

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        if len(polygon) < 3:
            return False
        inside = False
        prior = polygon[-1]
        for current in polygon:
            if ThreeSixtyFeatureEnricher._point_on_segment(point, prior, current):
                return True
            if (current[1] > point[1]) != (prior[1] > point[1]):
                crossing_x = (prior[0] - current[0]) * (
                    point[1] - current[1]
                ) / (prior[1] - current[1]) + current[0]
                if point[0] < crossing_x:
                    inside = not inside
            prior = current
        return inside

    @staticmethod
    def _point_on_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        cross = (point[1] - start[1]) * (end[0] - start[0]) - (
            point[0] - start[0]
        ) * (end[1] - start[1])
        if abs(cross) > 1e-9:
            return False
        return (
            min(start[0], end[0]) - 1e-9
            <= point[0]
            <= max(start[0], end[0]) + 1e-9
            and min(start[1], end[1]) - 1e-9
            <= point[1]
            <= max(start[1], end[1]) + 1e-9
        )

    @staticmethod
    def _goal_angle(x: float, y: float, goal_x: float) -> float:
        goal_distance_x = abs(goal_x - x)
        upper = math.atan2(
            SPADL_FIELD_WIDTH / 2.0 + GOAL_HALF_WIDTH_METRES - y,
            goal_distance_x,
        )
        lower = math.atan2(
            SPADL_FIELD_WIDTH / 2.0 - GOAL_HALF_WIDTH_METRES - y,
            goal_distance_x,
        )
        return abs(upper - lower)


def enrich_features_with_360(
    rows: tuple[ActionFeatureRow, ...],
    actions: tuple[SpadlAction, ...],
    inputs: SpadlInput,
) -> tuple[ActionFeatureRow, ...]:
    return ThreeSixtyFeatureEnricher().enrich(rows, actions, inputs)


__all__ = [
    "FEATURE_360_VERSION",
    "ThreeSixtyFeatureEnricher",
    "enrich_features_with_360",
]
