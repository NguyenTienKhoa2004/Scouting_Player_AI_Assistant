"""Create the default VAEP training artifacts with bounded memory."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.labeling_and_splitting.label_artifacts import (  # noqa: E402
    ChunkedTargetLabelWriter,
)
from pitchpulse.labeling_and_splitting.split_artifacts import (  # noqa: E402
    ChunkedSplitArtifactWriter,
)
from pitchpulse.model_dataset.artifacts import (  # noqa: E402
    ChunkedModelDatasetWriter,
)
from pitchpulse.model_dataset.finalize import (  # noqa: E402
    ChunkedPreparationFinalizer,
)
from pitchpulse.pipelines.settings import (  # noqa: E402
    DATASET_MANIFEST,
    FEATURE_OUTPUT,
    MODEL_OUTPUT,
    latest_artifact,
)


def main() -> None:
    def progress(message: str) -> None:
        print(message, flush=True)

    feature_directory = latest_artifact(FEATURE_OUTPUT, "manifest.json")
    labels = ChunkedTargetLabelWriter().write(
        feature_directory,
        dataset_manifest_path=DATASET_MANIFEST,
        output_root=MODEL_OUTPUT,
        progress=progress,
    )
    ChunkedSplitArtifactWriter().write(
        labels.directory,
        dataset_manifest_path=DATASET_MANIFEST,
        progress=progress,
    )
    ChunkedModelDatasetWriter().write(
        feature_directory,
        labels.directory,
        progress=progress,
    )
    result = ChunkedPreparationFinalizer().write(
        feature_directory,
        labels.directory,
        dataset_manifest_path=DATASET_MANIFEST,
        progress=progress,
    )
    print(f"Model dataset: {result.directory}", flush=True)


if __name__ == "__main__":
    main()
