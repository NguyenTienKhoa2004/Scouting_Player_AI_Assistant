"""Train the VAEP baselines, then tune and calibrate XGBoost models."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.model_training.logistic_baseline import (  # noqa: E402
    LogisticBaselineTrainer,
)
from pitchpulse.model_training.xgboost_models import XGBoostVaepTrainer  # noqa: E402
from pitchpulse.pipelines.settings import MODEL_OUTPUT, latest_artifact  # noqa: E402


def main() -> None:
    preparation = latest_artifact(MODEL_OUTPUT, "training_manifest.json")
    progress = lambda message: print(message, flush=True)
    manifest = json.loads(
        (preparation / "training_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("models", {}).get("status") == "not_trained":
        LogisticBaselineTrainer().write(preparation, progress=progress)
    result = XGBoostVaepTrainer().write(preparation, progress=progress)
    print(f"XGBoost models: {result.directory}")


if __name__ == "__main__":
    main()
