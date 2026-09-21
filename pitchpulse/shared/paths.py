"""Resolve repository paths shared by local pipeline entrypoints."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "compose.yaml").is_file()
            and (candidate / "pyproject.toml").is_file()
            and (candidate / "pitchpulse").is_dir()
        ):
            return candidate
    raise RuntimeError("could not locate the PitchPulse project root")


PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
BRONZE_ROOT = DATA_ROOT / "bronze"
STATSBOMB_BRONZE_ROOT = BRONZE_ROOT / "statsbomb-open-data"
STATSBOMB_BRONZE_DATA_ROOT = STATSBOMB_BRONZE_ROOT / "data"


__all__ = [
    "BRONZE_ROOT",
    "DATA_ROOT",
    "PROJECT_ROOT",
    "STATSBOMB_BRONZE_DATA_ROOT",
    "STATSBOMB_BRONZE_ROOT",
    "find_project_root",
]
