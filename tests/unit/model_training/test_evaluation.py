from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.model_training.evaluation import binary_probability_report  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
