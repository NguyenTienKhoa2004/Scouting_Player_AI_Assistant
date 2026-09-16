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
            and (candidate / "packages" / "matchmind" / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("could not locate the MatchMind project root")


PROJECT_ROOT = find_project_root()


__all__ = ["PROJECT_ROOT", "find_project_root"]
