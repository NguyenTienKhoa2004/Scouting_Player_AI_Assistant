"""Explicit entrypoint for the one-time frozen-model test evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.model_training.frozen_evaluation import (  # noqa: E402
    FrozenModelEvaluator,
)
from pitchpulse.pipelines.settings import MODEL_OUTPUT, latest_artifact  # noqa: E402


def main() -> None:
    preparation = latest_artifact(MODEL_OUTPUT, "training_manifest.json")
    progress = lambda message: print(message, flush=True)
    result = FrozenModelEvaluator().write(preparation, progress=progress)
    print(f"Test metrics: {result.test_metrics}")


if __name__ == "__main__":
    main()
