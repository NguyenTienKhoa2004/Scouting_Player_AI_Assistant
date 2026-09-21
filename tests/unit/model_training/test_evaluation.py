from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.model_training.evaluation import binary_probability_report  # noqa: E402
from pitchpulse.model_training.calibration import (  # noqa: E402
    fit_probability_calibrator,
)


class BinaryProbabilityReportTests(unittest.TestCase):
    def test_reports_probability_metrics_and_calibration(self) -> None:
        report = binary_probability_report(
            [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], inference_seconds=0.004
        )

        self.assertEqual(report["row_count"], 4)
        self.assertEqual(report["positive_count"], 2)
        self.assertEqual(report["roc_auc"], 1.0)
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(
            report["inference_latency"]["microseconds_per_row"], 1000.0
        )

    def test_fitted_calibrator_returns_bounded_probabilities(self) -> None:
        probabilities = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
        truth = [0, 0, 0, 1, 1, 1]

        calibrator = fit_probability_calibrator(
            probabilities, truth, random_seed=7
        )
        calibrated = calibrator.predict(probabilities)

        self.assertEqual(calibrator.method, "sigmoid")
        self.assertEqual(len(calibrated), len(probabilities))
        self.assertTrue(((calibrated >= 0.0) & (calibrated <= 1.0)).all())
        self.assertLess(calibrated[0], calibrated[-1])


if __name__ == "__main__":
    unittest.main()
