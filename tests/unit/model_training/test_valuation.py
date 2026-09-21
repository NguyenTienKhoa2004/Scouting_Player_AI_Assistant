from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.model_training.valuation import (  # noqa: E402
    ActionValuationError,
    ActionValueWriter,
    perspective_safe_values,
)


class PerspectiveSafeValueTests(unittest.TestCase):
    def test_writes_predictions_in_batches_and_preserves_row_groups(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        class IdentityCalibrator:
            @staticmethod
            def predict(values: object) -> object:
                return values

        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "dataset.parquet"
            output_path = Path(temporary) / "predictions.parquet"
            source = pa.table(
                {
                    "match_id": [1, 1, 2, 2, 3],
                    "action_id": [0, 1, 0, 1, 0],
                    "split": ["train", "train", "validation", "test", "test"],
                    "feature": [0.1, 0.2, 0.3, 0.4, 0.5],
                }
            )
            with pq.ParquetWriter(source_path, source.schema) as writer:
                writer.write_table(source.slice(0, 3), row_group_size=3)
                writer.write_table(source.slice(3, 2), row_group_size=2)

            def fake_predict(model: str, matrix: object) -> np.ndarray:
                probability = 0.25 if model == "score model" else 0.40
                return np.full(matrix.num_row(), probability)

            with patch(
                "pitchpulse.model_training.valuation.predict",
                side_effect=fake_predict,
            ):
                counts = ActionValueWriter._write_predictions(
                    dataset_path=source_path,
                    output_path=output_path,
                    features=("feature",),
                    models={
                        "scores": "score model",
                        "concedes": "concede model",
                    },
                    calibrators={
                        "scores": IdentityCalibrator(),
                        "concedes": IdentityCalibrator(),
                    },
                    batch_size=2,
                    n_jobs=1,
                    progress=None,
                )

            predictions = pq.ParquetFile(output_path)
            self.assertEqual(predictions.num_row_groups, 2)
            self.assertEqual(counts, {"train": 2, "validation": 1, "test": 2})
            table = predictions.read()
            np.testing.assert_allclose(table["p_score_after"], 0.25)
            np.testing.assert_allclose(table["p_concede_after"], 0.40)
            predictions.close()

    def test_matches_pinned_perspective_and_restart_semantics(self) -> None:
        actions = pd.DataFrame(
            {
                "match_id": [1] * 8,
                "team_id": [10, 10, 20, 20, 20, 10, 10, 10],
                "time_seconds": [0.0, 5.0, 6.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                "type_name": [
                    "pass",
                    "pass",
                    "pass",
                    "pass",
                    "shot",
                    "pass",
                    "shot_penalty",
                    "corner_short",
                ],
                "result_name": [
                    "success",
                    "success",
                    "success",
                    "success",
                    "success",
                    "success",
                    "success",
                    "success",
                ],
            }
        )
        scores = np.array([0.10, 0.20, 0.30, 0.25, 0.99, 0.05, 0.80, 0.10])
        concedes = np.array([0.20, 0.10, 0.40, 0.15, 0.01, 0.05, 0.02, 0.03])

        values = perspective_safe_values(actions, scores, concedes)

        np.testing.assert_allclose(
            values["p_score_before"],
            [0.10, 0.10, 0.10, 0.0, 0.25, 0.0, 0.792453, 0.0465],
        )
        np.testing.assert_allclose(
            values["p_concede_before"],
            [0.20, 0.20, 0.20, 0.0, 0.15, 0.0, 0.05, 0.0],
        )
        np.testing.assert_allclose(
            values["vaep_value"],
            values["offensive_value"] + values["defensive_value"],
        )
        self.assertAlmostEqual(values.loc[2, "offensive_value"], 0.20)
        self.assertAlmostEqual(values.loc[2, "defensive_value"], -0.20)

    def test_rejects_cross_match_formula_input(self) -> None:
        actions = pd.DataFrame(
            {
                "match_id": [1, 2],
                "team_id": [10, 10],
                "time_seconds": [0.0, 1.0],
                "type_name": ["pass", "pass"],
                "result_name": ["success", "success"],
            }
        )
        with self.assertRaises(ActionValuationError):
            perspective_safe_values(actions, [0.1, 0.2], [0.2, 0.1])

    def test_own_goal_keeps_the_pinned_library_perspective_rule(self) -> None:
        actions = pd.DataFrame(
            {
                "match_id": [1, 1],
                "team_id": [10, 20],
                "time_seconds": [1.0, 2.0],
                "type_name": ["pass", "pass"],
                "result_name": ["owngoal", "success"],
            }
        )

        values = perspective_safe_values(actions, [0.1, 0.3], [0.2, 0.4])

        # socceraction 1.5.3 flips team perspective but only applies the
        # post-goal reset to successful shot action types, not own-goal rows.
        self.assertAlmostEqual(values.loc[1, "p_score_before"], 0.2)
        self.assertAlmostEqual(values.loc[1, "p_concede_before"], 0.1)

    def test_aligns_eligible_predictions_as_action_subsequence(self) -> None:
        import pyarrow as pa

        actions = pa.table(
            {
                "match_id": [1, 1, 1],
                "action_id": [0, 1, 2],
                "player_id": [11, 12, 13],
                "team_id": [10, 10, 20],
                "period_id": [1, 1, 5],
                "time_seconds": [0.0, 1.0, 2.0],
                "type_name": ["pass", "pass", "shot_penalty"],
                "result_name": ["success", "success", "success"],
                "mapping_version": ["mapping"] * 3,
                "state_version": ["state"] * 3,
            }
        )
        predictions = pa.table(
            {
                "match_id": [1, 1],
                "action_id": [0, 1],
                "split": ["test", "test"],
                "p_score_after": [0.1, 0.2],
                "p_concede_after": [0.2, 0.1],
            }
        )

        aligned = ActionValueWriter._align_action_rows(actions, predictions, pa)

        self.assertEqual(aligned["action_id"].to_pylist(), [0, 1])
        self.assertEqual(aligned["player_id"].to_pylist(), [11, 12])


if __name__ == "__main__":
    unittest.main()
