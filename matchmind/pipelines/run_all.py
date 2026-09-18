"""Run the complete MatchMind pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from matchmind.pipelines.settings import (  # noqa: E402
    CORPUS_MANIFEST,
    FEATURE_OUTPUT,
)
from matchmind.shared.paths import PROJECT_ROOT  # noqa: E402


STEPS = (
    (
        "Ingesting data",
        "matchmind.ingestion.run",
        ("--corpus-manifest", str(CORPUS_MANIFEST)),
    ),
    (
        "Building features",
        "matchmind.vaep_features.run",
        (
            "--corpus-manifest",
            str(CORPUS_MANIFEST),
            "--output-dir",
            str(FEATURE_OUTPUT),
        ),
    ),
    ("Preparing model dataset", "matchmind.model_dataset.run", ()),
    ("Training models", "matchmind.model_training.run", ()),
)


def main() -> None:
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)}
    for number, (name, module, arguments) in enumerate(STEPS, start=1):
        print(f"{number}/4 {name}...", flush=True)
        subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    print("MatchMind pipeline completed successfully.")


if __name__ == "__main__":
    main()
