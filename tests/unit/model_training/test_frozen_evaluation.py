from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.model_training.calibration import ProbabilityCalibrator  # noqa: E402
from pitchpulse.model_training.frozen_evaluation import (  # noqa: E402
    FrozenModelEvaluator,
)
from pitchpulse.model_training.training_files import (  # noqa: E402
    TrainingArtifactError,
)
from pitchpulse.shared.file_io import file_sha256  # noqa: E402
from pitchpulse.vaep_features.feature_dataset_loader import (  # noqa: E402
    baseline_feature_allowlist,
)


class FrozenModelEvaluatorTests(unittest.TestCase):
    def test_evaluates_test_once_and_records_reports(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as raw:
            root = Path(raw)
            self._build_bundle(root)

            paths = FrozenModelEvaluator().write(
                root,
                batch_size=17,
                n_jobs=1,
            )

            report = json.loads(paths.test_metrics.read_text(encoding="utf-8"))
            self.assertTrue(report["test_split_accessed"])
            self.assertEqual(set(report["targets"]), {"scores", "concedes"})
            self.assertTrue(report["quality_gate"]["passed"])
            self.assertTrue(paths.calibration_report.is_file())
            self.assertTrue(paths.sanity_checks.is_file())
            self.assertTrue(paths.error_analysis.is_file())

            manifest = json.loads(paths.training_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "test_evaluation_complete")
            self.assertEqual(manifest["models"]["status"], "frozen_test_evaluated")
            self.assertTrue(manifest["production_promotion"]["allowed"])
            self.assertIn("test_metrics.json", manifest["artifacts"])

            # A completed bundle is verified and returned without reopening test data.
            (root / "model_dataset.parquet").unlink()
            repeated = FrozenModelEvaluator().write(root, batch_size=17, n_jobs=1)
            self.assertEqual(repeated.test_metrics, paths.test_metrics)

            with (root / "score_xgboost.json").open("ab") as target:
                target.write(b"post-evaluation drift")
            with self.assertRaises(TrainingArtifactError):
                FrozenModelEvaluator().write(root, batch_size=17, n_jobs=1)

    def test_rejects_a_changed_frozen_model(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as raw:
            root = Path(raw)
            self._build_bundle(root)
            with (root / "score_xgboost.json").open("ab") as target:
                target.write(b"drift")

            with self.assertRaises(TrainingArtifactError):
                FrozenModelEvaluator().write(root, batch_size=17, n_jobs=1)

    def test_isotonic_artifact_is_compatible_with_runtime_wrapper(self) -> None:
        estimator = IsotonicRegression(out_of_bounds="clip").fit(
            [0.1, 0.4, 0.8, 0.9],
            [0, 0, 1, 1],
        )
        calibrator = ProbabilityCalibrator("isotonic", estimator)
        predicted = calibrator.predict([0.2, 0.7])
        self.assertEqual(predicted.shape, (2,))
        self.assertTrue(((predicted >= 0) & (predicted <= 1)).all())

    @staticmethod
    def _build_bundle(root: Path) -> None:
        features = baseline_feature_allowlist()
        match_ids = np.repeat(np.arange(1, 5, dtype=np.int64), 30)
        action_ids = np.tile(np.arange(30, dtype=np.int64), 4)
        row_count = len(match_ids)
        scores = action_ids % 10 == 0
        concedes = action_ids % 15 == 1

        matrix = np.zeros((row_count, len(features)), dtype=np.float32)
        index = {name: position for position, name in enumerate(features)}
        matrix[:, index["actiontype_pass_a0"]] = 1
        matrix[scores, index["actiontype_pass_a0"]] = 0
        matrix[scores, index["actiontype_shot_a0"]] = 1
        matrix[concedes, index["actiontype_pass_a0"]] = 0
        matrix[concedes, index["actiontype_bad_touch_a0"]] = 1
        matrix[:, index["result_success_a0"]] = 1
        matrix[concedes, index["result_success_a0"]] = 0
        matrix[concedes, index["result_fail_a0"]] = 1
        matrix[:, index["dx_a0"]] = np.where(scores, 15.0, 1.0)
        matrix[:, index["goalscore_diff"]] = (action_ids % 3) - 1

        columns: dict[str, object] = {
            "match_id": pa.array(match_ids),
            "action_id": pa.array(action_ids),
        }
        for feature_index, name in enumerate(features):
            columns[name] = pa.array(matrix[:, feature_index])
        columns.update(
            {
                "scores": pa.array(scores),
                "concedes": pa.array(concedes),
                "split": pa.array(["test"] * row_count),
            }
        )
        dataset_path = root / "model_dataset.parquet"
        pq.write_table(pa.table(columns), dataset_path, row_group_size=19)

        assignments = pa.table(
            {
                "match_id": pa.array(np.arange(1, 5, dtype=np.int64)),
                "competition_id": pa.array([10, 10, 20, 20]),
                "season_id": pa.array([100, 100, 200, 200]),
                "split": pa.array(["test"] * 4),
            }
        )
        split_path = root / "split_assignments.parquet"
        pq.write_table(assignments, split_path)

        feature_path = root / "feature_allowlist.json"
        feature_path.write_text(
            json.dumps(
                {
                    "feature_columns": list(features),
                    "feature_allowlist_sha256": "f" * 64,
                }
            ),
            encoding="utf-8",
        )
        training_matrix = xgb.DMatrix(sparse.csr_matrix(matrix))
        model_paths: dict[str, Path] = {}
        calibration_paths: dict[str, Path] = {}
        for target, stem, truth in (
            ("scores", "score", scores),
            ("concedes", "concede", concedes),
        ):
            training_matrix.set_label(truth.astype(np.int8))
            model = xgb.train(
                {
                    "objective": "binary:logistic",
                    "tree_method": "hist",
                    "max_depth": 2,
                    "eta": 0.3,
                    "nthread": 1,
                    "seed": 7,
                },
                training_matrix,
                num_boost_round=15,
            )
            model_path = root / f"{stem}_xgboost.json"
            model.save_model(model_path)
            raw_probability = model.predict(training_matrix)
            estimator = IsotonicRegression(out_of_bounds="clip").fit(
                raw_probability,
                truth.astype(np.int8),
            )
            calibration_path = root / f"{stem}_calibration.joblib"
            joblib.dump(
                ProbabilityCalibrator("isotonic", estimator),
                calibration_path,
            )
            model_paths[target] = model_path
            calibration_paths[target] = calibration_path

        validation_path = root / "xgboost_validation_report.json"
        validation_path.write_text(
            json.dumps(
                {
                    "targets": {
                        target: {"beats_logistic_baseline": True}
                        for target in ("scores", "concedes")
                    }
                }
            ),
            encoding="utf-8",
        )
        artifact_paths = [
            dataset_path,
            split_path,
            feature_path,
            validation_path,
            *model_paths.values(),
            *calibration_paths.values(),
        ]
        artifacts = {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in artifact_paths
        }
        manifest = {
            "schema_version": 1,
            "stage": "xgboost_training",
            "status": "xgboost_validation_complete",
            "source": {"dataset_fingerprint": "d" * 64},
            "features": {"feature_allowlist_sha256": "f" * 64},
            "split": {
                "splits": {
                    "test": {
                        "eligible_label_count": row_count,
                        "match_count": 4,
                    }
                }
            },
            "models": {
                "status": "xgboost_validation_complete",
                "xgboost_bundle_version": "test-bundle-v1",
            },
            "production_promotion": {
                "allowed": False,
                "blockers": ["models:untouched_test_not_run"],
            },
            "artifacts": artifacts,
        }
        (root / "training_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
