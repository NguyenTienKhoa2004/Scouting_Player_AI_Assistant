from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.model_dataset import run  # noqa: E402


class ChunkedPreparationRunTests(unittest.TestCase):
    @patch.object(run, "ChunkedPreparationFinalizer")
    @patch.object(run, "ChunkedModelDatasetWriter")
    @patch.object(run, "ChunkedSplitArtifactWriter")
    @patch.object(run, "ChunkedTargetLabelWriter")
    @patch.object(run, "latest_artifact")
    def test_main_uses_the_chunked_pipeline(
        self,
        latest_artifact: Mock,
        label_writer_type: Mock,
        split_writer_type: Mock,
        dataset_writer_type: Mock,
        finalizer_type: Mock,
    ) -> None:
        feature_directory = Path("feature-artifact")
        label_directory = Path("label-artifact")
        latest_artifact.return_value = feature_directory
        label_writer_type.return_value.write.return_value = SimpleNamespace(
            directory=label_directory
        )
        finalizer_type.return_value.write.return_value = SimpleNamespace(
            directory=label_directory
        )

        run.main()

        latest_artifact.assert_called_once_with(run.FEATURE_OUTPUT, "manifest.json")
        label_writer_type.return_value.write.assert_called_once_with(
            feature_directory,
            dataset_manifest_path=run.DATASET_MANIFEST,
            output_root=run.MODEL_OUTPUT,
            progress=ANY,
        )
        split_writer_type.return_value.write.assert_called_once_with(
            label_directory,
            dataset_manifest_path=run.DATASET_MANIFEST,
            progress=ANY,
        )
        dataset_writer_type.return_value.write.assert_called_once_with(
            feature_directory,
            label_directory,
            progress=ANY,
        )
        finalizer_type.return_value.write.assert_called_once_with(
            feature_directory,
            label_directory,
            dataset_manifest_path=run.DATASET_MANIFEST,
            progress=ANY,
        )


if __name__ == "__main__":
    unittest.main()
