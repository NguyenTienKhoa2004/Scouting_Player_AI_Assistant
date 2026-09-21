from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.model_training.xgboost_contracts import (  # noqa: E402
    TARGETS,
    XGBoostTrainingError,
)
from pitchpulse.model_training.xgboost_data import validate_phase_targets  # noqa: E402
from pitchpulse.model_training.xgboost_training import (  # noqa: E402
    beats_baseline,
    candidate_rank,
)


class XGBoostTrainingComponentTests(unittest.TestCase):
    def test_candidate_rank_prefers_pr_auc_then_probability_quality(self) -> None:
        strong_pr_auc = {"pr_auc": 0.50, "log_loss": 0.60, "brier_score": 0.25}
        strong_log_loss = {"pr_auc": 0.49, "log_loss": 0.40, "brier_score": 0.18}
        self.assertGreater(
            candidate_rank(strong_pr_auc),
            candidate_rank(strong_log_loss),
        )

    def test_baseline_rule_requires_better_pr_auc_and_probability_quality(self) -> None:
        baseline = {"pr_auc": 0.40, "brier_score": 0.20, "log_loss": 0.50}
        self.assertTrue(
            beats_baseline(
                {"pr_auc": 0.41, "brier_score": 0.19, "log_loss": 0.55},
                baseline,
            )
        )
        self.assertFalse(
            beats_baseline(
                {"pr_auc": 0.41, "brier_score": 0.21, "log_loss": 0.51},
                baseline,
            )
        )

    def test_each_validation_phase_requires_both_target_classes(self) -> None:
        targets = {
            target: np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
            for target in TARGETS
        }
        phases = {
            "tuning": np.array([0, 1]),
            "calibration_fit": np.array([2, 3]),
            "calibration_evaluation": np.array([4, 5]),
        }
        validate_phase_targets(targets, phases)
        targets["scores"][4:] = 0
        with self.assertRaises(XGBoostTrainingError):
            validate_phase_targets(targets, phases)


if __name__ == "__main__":
    unittest.main()
