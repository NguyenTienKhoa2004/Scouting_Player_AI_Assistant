"""Run the complete PitchPulse pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.pipelines.settings import (  # noqa: E402
    DATASET_MANIFEST,
    FEATURE_OUTPUT,
)
from pitchpulse.shared.paths import PROJECT_ROOT  # noqa: E402


STEPS = (
    (
        "Ingesting data",
        "pitchpulse.ingestion.run",
        ("--dataset-manifest", str(DATASET_MANIFEST)),
    ),
    (
        "Building features",
        "pitchpulse.vaep_features.run",
        (
            "--dataset-manifest",
            str(DATASET_MANIFEST),
            "--output-dir",
            str(FEATURE_OUTPUT),
        ),
    ),
    (
        "Registering feature dataset",
        "pitchpulse.vaep_features.register_dataset",
        (),
    ),
    ("Preparing model dataset", "pitchpulse.model_dataset.run", ()),
    ("Training models", "pitchpulse.model_training.run", ()),
    (
        "Evaluating frozen models",
        "pitchpulse.model_training.run_test_evaluation",
        (),
    ),
    ("Calculating action VAEP", "pitchpulse.model_training.run_valuation", ()),
    (
        "Aggregating player VAEP",
        "pitchpulse.player_vaep.run",
        (),
    ),
    (
        "Persisting VAEP artifacts",
        "pitchpulse.model_training.run_persistence",
        (),
    ),
)


def main() -> None:
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)}
    for number, (name, module, arguments) in enumerate(STEPS, start=1):
        print(f"{number}/{len(STEPS)} {name}...", flush=True)
        subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    print("PitchPulse pipeline completed successfully.")


if __name__ == "__main__":
    main()
