"""Entrypoint for frozen-model inference and perspective-safe VAEP valuation."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.model_training.valuation import ActionValueWriter  # noqa: E402
from pitchpulse.pipelines.settings import MODEL_OUTPUT, latest_artifact  # noqa: E402


def main() -> None:
    preparation = latest_artifact(MODEL_OUTPUT, "training_manifest.json")
    progress = lambda message: print(message, flush=True)
    result = ActionValueWriter().write(preparation, progress=progress)
    print(f"Action values: {result.action_values}")


if __name__ == "__main__":
    main()
