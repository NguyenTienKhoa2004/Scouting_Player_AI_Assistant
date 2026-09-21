from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.pipelines import run_all  # noqa: E402


class RunAllPipelineTests(unittest.TestCase):
    def test_runs_the_complete_pipeline_in_order(self) -> None:
        expected_modules = [
            "pitchpulse.ingestion.run",
            "pitchpulse.vaep_features.run",
            "pitchpulse.vaep_features.register_corpus",
            "pitchpulse.model_dataset.run",
            "pitchpulse.model_training.run",
            "pitchpulse.model_training.run_test_evaluation",
            "pitchpulse.model_training.run_valuation",
            "pitchpulse.player_vaep.run",
            "pitchpulse.model_training.run_persistence",
        ]

        with patch("pitchpulse.pipelines.run_all.subprocess.run") as run:
            run_all.main()

        self.assertEqual(len(run.call_args_list), len(expected_modules))
        actual_modules = [item.args[0][2] for item in run.call_args_list]
        self.assertEqual(actual_modules, expected_modules)
        for item in run.call_args_list:
            self.assertEqual(item.kwargs["cwd"], run_all.PROJECT_ROOT)
            self.assertTrue(item.kwargs["check"])
            self.assertEqual(
                item.kwargs["env"]["PYTHONPATH"], str(run_all.REPOSITORY_ROOT)
            )


if __name__ == "__main__":
    unittest.main()
