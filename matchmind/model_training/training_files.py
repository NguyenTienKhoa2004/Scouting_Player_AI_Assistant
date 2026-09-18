"""Small file helpers shared by model-training workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from matchmind.shared.file_io import read_json, write_json


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


__all__ = ["find_training_file", "read_json", "write_json"]
