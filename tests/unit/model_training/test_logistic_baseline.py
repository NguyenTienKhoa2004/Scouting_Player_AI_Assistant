from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.vaep_features.feature_builder import BASE_FEATURE_VERSION  # noqa: E402
from pitchpulse.vaep_features.feature_dataset_loader import baseline_feature_allowlist  # noqa: E402
from pitchpulse.model_dataset.feature_allowlist import (  # noqa: E402
    feature_allowlist_manifest,
)
from pitchpulse.model_training.logistic_baseline import LogisticBaselineTrainer  # noqa: E402


class LogisticBaselineTrainerTests(unittest.TestCase):
    def test_trains_both_targets_and_evaluates_validation_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as raw:
            root = Path(raw)
            allowlist = baseline_feature_allowlist()
            row_count = 180
            splits = ["train"] * 100 + ["validation"] * 50 + ["test"] * 30
            values: dict[str, object] = {
                "scores": pa.array([index % 11 == 0 for index in range(row_count)]),
                "concedes": pa.array([index % 17 == 0 for index in range(row_count)]),
                "split": pa.array(splits),
            }
            for feature_index, name in enumerate(allowlist):
                values[name] = pa.array(
                    [float((row + feature_index) % 7) for row in range(row_count)]
                )
            dataset = pa.table(values)
            dataset_path = root / "model_dataset.parquet"
            pq.write_table(dataset, dataset_path, row_group_size=23)
            feature_manifest = feature_allowlist_manifest(
                allowlist,
                source_dataset_fingerprint="a" * 64,
                feature_version=BASE_FEATURE_VERSION,
            )
            (root / "feature_allowlist.json").write_text(
                json.dumps(feature_manifest), encoding="utf-8"
            )
            training_manifest = {
                "stage": "data_preparation",
                "status": "ready_for_training",
                "artifacts": {
                    "model_dataset.parquet": {
                        "path": str(dataset_path),
                    }
                },
                "models": {"status": "not_trained"},
            }
            (root / "training_manifest.json").write_text(
                json.dumps(training_manifest), encoding="utf-8"
            )

            paths = LogisticBaselineTrainer().write(
                root, epochs=2, batch_size=19, random_seed=123
            )

            report = json.loads(paths.report.read_text(encoding="utf-8"))
            self.assertEqual(report["evaluation_split"], "validation")
            self.assertFalse(report["test_split_accessed"])
            self.assertEqual(report["metrics"]["scores"]["row_count"], 50)
            self.assertEqual(report["metrics"]["concedes"]["row_count"], 50)
            self.assertEqual(
                report["class_weights"]["scores"]["negative_count"]
                + report["class_weights"]["scores"]["positive_count"],
                100,
            )
            self.assertTrue(paths.preprocessor.is_file())
            self.assertTrue(paths.score_model.is_file())
            self.assertTrue(paths.concede_model.is_file())
            updated = json.loads(paths.training_manifest.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "logistic_baselines_evaluated")
            self.assertEqual(
                updated["models"]["status"], "logistic_baselines_evaluated"
            )
            self.assertTrue(updated["ready_for_xgboost"])


if __name__ == "__main__":
    unittest.main()
