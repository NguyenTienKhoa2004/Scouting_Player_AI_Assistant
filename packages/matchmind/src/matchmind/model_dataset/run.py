"""Create the default VAEP labels, splits, and model dataset."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_SRC))

from matchmind.vaep_features.artifact_loader import FeatureArtifactLoader  # noqa: E402
from matchmind.model_dataset.preparation import VaepDataPreparationWriter  # noqa: E402
from matchmind.pipelines.settings import (  # noqa: E402
    CORPUS_MANIFEST,
    FEATURE_OUTPUT,
    MODEL_OUTPUT,
    latest_artifact,
)


def main() -> None:
    features = FeatureArtifactLoader().load(
        latest_artifact(FEATURE_OUTPUT, "manifest.json")
    )
    result = VaepDataPreparationWriter().write(
        features,
        corpus_manifest_path=CORPUS_MANIFEST,
        output_root=MODEL_OUTPUT,
    )
    print(f"Model dataset: {result.directory}")


if __name__ == "__main__":
    main()
