"""Train and evaluate the default VAEP logistic baseline models."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_SRC))

from matchmind.ml import LogisticBaselineTrainer  # noqa: E402
from matchmind.pipelines.settings import MODEL_OUTPUT, latest_artifact  # noqa: E402


def main() -> None:
    preparation = latest_artifact(MODEL_OUTPUT, "training_manifest.json")
    result = LogisticBaselineTrainer().write(
        preparation,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Models: {result.directory}")


if __name__ == "__main__":
    main()
