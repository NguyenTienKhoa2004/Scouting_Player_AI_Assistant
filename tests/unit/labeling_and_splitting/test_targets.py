from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyarrow as pa
from socceraction.spadl import config as spadlconfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.labeling_and_splitting.targets import (  # noqa: E402
    PENALTY_SHOOTOUT_EXCLUSION,
    TARGET_POLICY_VERSION,
    TargetLabelBuilder,
    baseline_target_policy,
)


class TargetLabelBuilderTests(unittest.TestCase):
    def test_shootout_is_removed_before_labels_but_retained_for_audit(self) -> None:
        actions = self._actions(
            [
                self._row(0, team_id=1),
                self._row(
                    1,
                    team_id=1,
                    action_type="shot_penalty",
                    result="success",
                    period_id=5,
                    is_shootout=True,
                ),
            ]
        )

        result = TargetLabelBuilder().build(actions, analytics_run_id=7)
        labels = result.labels.to_pylist()

        self.assertFalse(labels[0]["scores"])
        self.assertTrue(labels[0]["eligible"])
        self.assertIsNone(labels[1]["scores"])
        self.assertIsNone(labels[1]["concedes"])
        self.assertFalse(labels[1]["eligible"])
        self.assertEqual(
            labels[1]["exclusion_reason"], PENALTY_SHOOTOUT_EXCLUSION
        )
        self.assertEqual(result.audit["counts"]["excluded_count"], 1)
        self.assertTrue(result.audit["checks"]["shootouts"]["all_labels_null"])

    def test_target_window_is_current_action_through_nine_following(self) -> None:
        rows = [self._row(index, team_id=1) for index in range(11)]
        rows[10] = self._row(
            10, team_id=1, action_type="shot", result="success"
        )

        labels = TargetLabelBuilder().build(
            self._actions(rows), analytics_run_id=None
        ).labels.to_pylist()

        self.assertFalse(labels[0]["scores"])
        self.assertTrue(labels[1]["scores"])
        self.assertTrue(labels[10]["scores"])

    def test_target_window_stops_at_match_boundary(self) -> None:
        actions = self._actions(
            [
                self._row(0, match_id=101, team_id=1),
                self._row(
                    0,
                    match_id=202,
                    team_id=1,
                    action_type="shot",
                    result="success",
                ),
            ]
        )

        labels = TargetLabelBuilder().build(
            actions, analytics_run_id=None
        ).labels.to_pylist()

        self.assertFalse(labels[0]["scores"])
        self.assertTrue(labels[1]["scores"])

    def test_own_goal_uses_socceraction_acting_team_perspective(self) -> None:
        actions = self._actions(
            [
                self._row(0, team_id=1),
                self._row(
                    1, team_id=1, action_type="shot", result="owngoal"
                ),
            ]
        )

        result = TargetLabelBuilder().build(actions, analytics_run_id=None)
        labels = result.labels.to_pylist()

        self.assertTrue(labels[0]["concedes"])
        self.assertTrue(labels[1]["concedes"])
        self.assertFalse(labels[1]["scores"])
        self.assertEqual(
            result.audit["checks"]["own_goals"][
                "current_action_concedes_mismatches"
            ],
            0,
        )

    def test_policy_is_explicit_and_versioned(self) -> None:
        policy = baseline_target_policy()

        self.assertEqual(policy["target_policy_version"], TARGET_POLICY_VERSION)
        self.assertEqual(policy["window"]["start_offset"], 0)
        self.assertEqual(policy["window"]["end_offset_inclusive"], 9)
        self.assertEqual(
            policy["exclusions"][PENALTY_SHOOTOUT_EXCLUSION]["rule"],
            "exclude_before_label_generation",
        )

    @staticmethod
    def _row(
        action_id: int,
        *,
        match_id: int = 101,
        team_id: int,
        action_type: str = "pass",
        result: str = "success",
        period_id: int = 1,
        is_shootout: bool = False,
    ) -> dict[str, object]:
        return {
            "match_id": match_id,
            "action_id": action_id,
            "period_id": period_id,
            "team_id": team_id,
            "type_id": spadlconfig.actiontypes.index(action_type),
            "type_name": action_type,
            "result_id": spadlconfig.results.index(result),
            "result_name": result,
            "bodypart_id": spadlconfig.bodyparts.index("foot"),
            "is_shootout": is_shootout,
            "possession_changed": False,
        }

    @staticmethod
    def _actions(rows: list[dict[str, object]]) -> pa.Table:
        return pa.Table.from_pylist(rows)


if __name__ == "__main__":
    unittest.main()
