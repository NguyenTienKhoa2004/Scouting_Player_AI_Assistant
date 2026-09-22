"""Shared paths for the default PitchPulse pipelines."""

from pathlib import Path

from pitchpulse.shared.paths import PROJECT_ROOT


DATASET_MANIFEST = (
    PROJECT_ROOT / "configs" / "datasets" / "vaep-training-dataset-v1.json"
)
FEATURE_OUTPUT = PROJECT_ROOT / "artifacts" / "features" / "plan03"
MODEL_OUTPUT = PROJECT_ROOT / "artifacts" / "models" / "plan04"


def latest_artifact(root: Path, marker: str) -> Path:
    manifests = list(root.rglob(marker))
    if not manifests:
        raise FileNotFoundError(f"No {marker} found under {root}")
    return max(manifests, key=lambda path: path.stat().st_mtime_ns).parent
