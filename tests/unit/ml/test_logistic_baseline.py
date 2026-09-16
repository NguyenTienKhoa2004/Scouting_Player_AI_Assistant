from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.analytics.features.vaep_features import BASE_FEATURE_VERSION  # noqa: E402
from matchmind.ml.feature_artifacts import baseline_feature_allowlist  # noqa: E402
from matchmind.ml.dataset import feature_allowlist_manifest  # noqa: E402
from matchmind.ml.logistic_baseline import LogisticBaselineTrainer  # noqa: E402


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
                        "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                    }
                },
                "models": {"status": "not_trained"},
                "production_promotion": {
                    "corpus_adequate": True,
                    "allowed": False,
                    "blockers": ["models:not_trained_or_evaluated"],
                },
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
            self.assertFalse(updated["production_promotion"]["allowed"])


if __name__ == "__main__":
    unittest.main()
