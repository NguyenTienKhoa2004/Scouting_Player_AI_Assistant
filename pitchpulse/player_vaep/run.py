"""Entrypoint for player minutes and VAEP aggregation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.player_vaep.artifacts import (  # noqa: E402
    PlayerAggregationWriter,
)
from pitchpulse.pipelines.settings import (  # noqa: E402
    DATASET_MANIFEST,
    MODEL_OUTPUT,
    latest_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate action VAEP by player")
    parser.add_argument(
        "--minimum-minutes",
        type=float,
        default=450.0,
        help="Minimum minutes required for ranking eligibility (default: 450)",
    )
    args = parser.parse_args()

    artifact_directory = latest_artifact(MODEL_OUTPUT, "training_manifest.json")
    progress = lambda message: print(message, flush=True)
    result = PlayerAggregationWriter().write(
        artifact_directory,
        DATASET_MANIFEST,
        minimum_minutes=args.minimum_minutes,
        progress=progress,
    )
    print(f"Player VAEP: {result.player_vaep}")


if __name__ == "__main__":
    main()
