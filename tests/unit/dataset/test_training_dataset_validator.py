from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.dataset.training_dataset_validator import (  # noqa: E402
    TRAINING_DATASET_SCHEMA_VERSION,
    TrainingDatasetError,
    load_training_dataset_manifest,
)
from pitchpulse.dataset.bronze import (  # noqa: E402
    BronzeSourceError,
    validate_bronze_repository,
)
from pitchpulse.model_dataset.training_manifest import (  # noqa: E402
    TrainingDatasetNotReady,
    require_training_dataset_ready,
)


DATASET_MANIFEST = (
    PROJECT_ROOT / "configs" / "datasets" / "vaep-training-dataset-v1.json"
)


class TrainingDatasetValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_training_dataset_manifest(DATASET_MANIFEST)

    def test_pinned_dataset_reconciles_with_local_statsbomb_data(self) -> None:
        self.assertEqual(
            self.dataset.dataset_id,
            "statsbomb-open-data-male-multicompetition-v1",
        )
        self.assertEqual(len(self.dataset.matches), 1831)
        self.assertEqual(len(self.dataset.match_ids), 1831)
        self.assertEqual(len(self.dataset.selections), 10)
        self.assertEqual(self.dataset.scope["competition_gender"], "male")
        self.assertFalse(self.dataset.scope["three_sixty_required"])
        self.assertEqual(
            len({match.competition_id for match in self.dataset.matches}), 8
        )
        self.assertEqual(TRAINING_DATASET_SCHEMA_VERSION, 3)
        self.assertIsNotNone(self.dataset.bronze)
        assert self.dataset.bronze is not None
        self.assertEqual(self.dataset.bronze.source_format, "statsbomb-open-data-json")
        self.assertTrue(self.dataset.bronze.immutable)
        self.assertEqual(
            self.dataset.bronze.data_root,
            PROJECT_ROOT / "data" / "bronze" / "statsbomb-open-data" / "data",
        )
        self.assertTrue(
            all(
                selection.data_root == self.dataset.bronze.data_root
                for selection in self.dataset.selections
            )
        )

    def test_bronze_repository_matches_pinned_source_commit(self) -> None:
        assert self.dataset.bronze is not None
        report = validate_bronze_repository(
            self.dataset.bronze.data_root,
            str(self.dataset.source["git_commit"]),
        )

        self.assertTrue(report.tracked_tree_clean)
        self.assertEqual(report.actual_commit, report.expected_commit)

    def test_bronze_repository_rejects_wrong_commit(self) -> None:
        expected = "a" * 40
        with patch(
            "pitchpulse.dataset.bronze._git",
            return_value="b" * 40,
        ):
            with self.assertRaisesRegex(BronzeSourceError, "commit mismatch"):
                validate_bronze_repository(PROJECT_ROOT / "data", expected)

    def test_bronze_repository_rejects_local_changes(self) -> None:
        expected = "a" * 40
        with patch(
            "pitchpulse.dataset.bronze._git",
            side_effect=[expected, " M data/events/1.json"],
        ):
            with self.assertRaisesRegex(BronzeSourceError, "local changes"):
                validate_bronze_repository(PROJECT_ROOT / "data", expected)

    def test_manifest_rejects_selection_path_outside_bronze(self) -> None:
        manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            directory = Path(temporary)
            bronze_root = PROJECT_ROOT / "data" / "bronze" / "statsbomb-open-data" / "data"
            manifest["bronze"]["data_root"] = os.path.relpath(
                bronze_root,
                directory,
            )
            manifest["source"]["license_file"] = os.path.relpath(
                bronze_root.parent / "LICENSE.pdf",
                directory,
            )
            manifest["selections"][0]["matches_path"] = "../LICENSE.pdf"
            path = directory / "dataset.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(TrainingDatasetError, "escapes bronze.data_root"):
                load_training_dataset_manifest(path)

    def test_complete_dataset_with_adequate_splits_passes_gate(self) -> None:
        assessment = self.dataset.assess(
            self.dataset.match_ids,
            self._split_manifest(matches=300, positives=500),
        )

        self.assertTrue(assessment["adequate_for_production_evaluation"])
        self.assertEqual(assessment["blockers"], [])
        self.assertEqual(assessment["materialized"]["coverage_rate"], 1.0)

    def test_world_cup_only_artifact_remains_experimental(self) -> None:
        world_cup_2022_ids = {
            match.match_id
            for match in self.dataset.matches
            if match.competition_id == 43 and match.season_id == 106
        }
        assessment = self.dataset.assess(
            world_cup_2022_ids,
            self._split_manifest(matches=10, positives=50),
        )

        self.assertFalse(assessment["adequate_for_production_evaluation"])
        self.assertEqual(assessment["materialized"]["match_count"], 64)
        self.assertIn("minimum_matches", assessment["blockers"])
        self.assertIn("complete_declared_dataset", assessment["blockers"])
        self.assertIn("test.minimum_concedes_positive", assessment["blockers"])

    def test_training_guard_rejects_inadequate_dataset(self) -> None:
        manifest = {
            "ready_for_training": False,
            "training_blockers": ["minimum_matches"],
        }

        with self.assertRaisesRegex(
            TrainingDatasetNotReady, "not ready"
        ):
            require_training_dataset_ready(manifest)

    @staticmethod
    def _split_manifest(*, matches: int, positives: int) -> dict[str, object]:
        return {
            "splits": {
                split: {
                    "match_count": matches,
                    "scores_positive_count": positives,
                    "concedes_positive_count": positives,
                }
                for split in ("train", "validation", "test")
            }
        }


if __name__ == "__main__":
    unittest.main()
