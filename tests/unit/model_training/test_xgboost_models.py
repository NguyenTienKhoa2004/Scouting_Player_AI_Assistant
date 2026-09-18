from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from matchmind.model_dataset.builder import feature_allowlist_manifest  # noqa: E402
from matchmind.model_training.logistic_baseline import LogisticBaselineTrainer  # noqa: E402
from matchmind.model_training.xgboost_models import (  # noqa: E402
    XGBoostTrainingError,
    XGBoostVaepTrainer,
)
from matchmind.model_training.xgboost_data import load_split  # noqa: E402
from matchmind.vaep_features.feature_dataset_loader import baseline_feature_allowlist  # noqa: E402
from matchmind.vaep_features.feature_builder import BASE_FEATURE_VERSION  # noqa: E402


class XGBoostVaepTrainerTests(unittest.TestCase):
    CANDIDATES = (
        {
            "max_depth": 1,
            "learning_rate": 0.2,
            "n_estimators": 10,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        {
            "max_depth": 2,
            "learning_rate": 0.2,
            "n_estimators": 12,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    )

    def test_trains_tunes_and_calibrates_both_targets_without_test(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as raw:
            root = Path(raw)
            allowlist = baseline_feature_allowlist()
            rows_per_match = 20
            split_matches = {
                "train": list(range(1, 13)),
                "validation": list(range(13, 25)),
                "test": list(range(25, 28)),
            }
            ordered_matches = [
                (match_id, split)
                for split, match_ids in split_matches.items()
                for match_id in match_ids
            ]
            match_ids = [
                match_id
                for match_id, _ in ordered_matches
                for _ in range(rows_per_match)
            ]
            splits = [
                split
                for _, split in ordered_matches
                for _ in range(rows_per_match)
            ]
            row_count = len(match_ids)
            scores = [row % rows_per_match in {0, 7, 14} for row in range(row_count)]
            concedes = [row % rows_per_match in {1, 12} for row in range(row_count)]
            values: dict[str, object] = {
                "match_id": pa.array(match_ids, type=pa.int64()),
                "scores": pa.array(scores),
                "concedes": pa.array(concedes),
                "split": pa.array(splits),
            }
            for feature_index, name in enumerate(allowlist):
                if feature_index == 0:
                    feature_values = [float(value) for value in scores]
                elif feature_index == 1:
                    feature_values = [float(value) for value in concedes]
                else:
                    feature_values = [
                        float((row + feature_index) % 29 == 0)
                        for row in range(row_count)
                    ]
                values[name] = pa.array(feature_values, type=pa.float32())
            dataset_path = root / "model_dataset.parquet"
            pq.write_table(pa.table(values), dataset_path, row_group_size=37)

            assignment_rows = []
            for split_order, (match_id, split) in enumerate(ordered_matches):
                assignment_rows.append(
                    {
                        "match_id": match_id,
                        "split": split,
                        "split_order": split_order,
                    }
                )
            split_path = root / "split_assignments.parquet"
            pq.write_table(pa.Table.from_pylist(assignment_rows), split_path)

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
                    dataset_path.name: self._artifact(dataset_path),
                    split_path.name: self._artifact(split_path),
                },
                "models": {"status": "not_trained"},
            }
            (root / "training_manifest.json").write_text(
                json.dumps(training_manifest), encoding="utf-8"
            )
            LogisticBaselineTrainer().write(
                root,
                epochs=1,
                batch_size=31,
                random_seed=123,
            )

            paths = XGBoostVaepTrainer().write(
                root,
                candidates=self.CANDIDATES,
                batch_size=31,
                random_seed=321,
                early_stopping_rounds=3,
                n_jobs=1,
            )

            report = json.loads(paths.report.read_text(encoding="utf-8"))
            self.assertFalse(report["test_split_accessed"])
            self.assertEqual(set(report["targets"]), {"scores", "concedes"})
            score_report = report["targets"]["scores"]
            self.assertEqual(len(score_report["candidate_results"]), 2)
            selected = score_report["selected_candidate"]
            self.assertEqual(
                score_report["parameters"],
                score_report["candidate_results"][selected]["parameters"],
            )
            validation_rows = sum(
                phase["row_count"]
                for phase in report["data"]["validation_partitions"].values()
            )
            self.assertEqual(validation_rows, 12 * rows_per_match)
            self.assertTrue(paths.score_model.is_file())
            self.assertTrue(paths.concede_model.is_file())
            score_calibrator = joblib.load(paths.score_calibration)
            calibrated = score_calibrator.predict([0.1, 0.9])
            self.assertTrue(all(0.0 <= value <= 1.0 for value in calibrated))

            updated = json.loads(paths.training_manifest.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "xgboost_validation_complete")
            self.assertEqual(
                updated["models"]["status"], "xgboost_validation_complete"
            )
            self.assertIn("ready_for_test_evaluation", updated)

    def test_refuses_to_load_the_test_split(self) -> None:
        with self.assertRaises(XGBoostTrainingError):
            load_split(
                Path("unused.parquet"),
                ("unused",),
                "test",
                batch_size=10,
            )

    @staticmethod
    def _artifact(path: Path) -> dict[str, str]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


if __name__ == "__main__":
    unittest.main()
