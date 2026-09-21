"""Small file helpers shared by model-training workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pitchpulse.shared.file_io import file_sha256, read_json, write_json


class TrainingArtifactError(ValueError):
    """Raised when a declared training artifact is absent or has drifted."""


def find_training_file(
    root: Path,
    manifest: Mapping[str, Any],
    filename: str,
) -> Path:
    record = (manifest.get("artifacts") or {}).get(filename, {})
    declared = Path(str(record.get("path", "")))
    if declared and not declared.is_absolute():
        declared = root / declared
    if declared.is_file():
        return declared

    direct_path = root / filename
    if direct_path.is_file():
        return direct_path

    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {filename} below {root}, found {len(matches)}"
        )
    return matches[0]


def verify_training_file(
    root: Path,
    manifest: Mapping[str, Any],
    filename: str,
) -> Path:
    """Resolve an artifact and require its declared SHA-256 to match."""

    record = (manifest.get("artifacts") or {}).get(filename)
    if not isinstance(record, Mapping):
        raise TrainingArtifactError(
            f"Training manifest does not declare artifact {filename}"
        )
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise TrainingArtifactError(
            f"Training manifest lacks a valid SHA-256 for {filename}"
        )
    path = find_training_file(root, manifest, filename)
    actual = file_sha256(path)
    if actual != expected:
        raise TrainingArtifactError(
            f"Training artifact hash mismatch for {filename}: "
            f"expected {expected}, got {actual}"
        )
    return path


__all__ = [
    "TrainingArtifactError",
    "find_training_file",
    "read_json",
    "verify_training_file",
    "write_json",
]
