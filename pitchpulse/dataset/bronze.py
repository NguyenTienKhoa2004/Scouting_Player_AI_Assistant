"""Integrity checks for the immutable Bronze source repository."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class BronzeSourceError(ValueError):
    """Raised when the local Bronze snapshot does not match its source pin."""


@dataclass(frozen=True, slots=True)
class BronzeSourceReport:
    repository_root: Path
    expected_commit: str
    actual_commit: str
    tracked_tree_clean: bool


def validate_bronze_repository(
    data_root: Path, expected_commit: str
) -> BronzeSourceReport:
    """Verify that Bronze is the clean working tree of the pinned Git commit."""

    if not _GIT_COMMIT_PATTERN.fullmatch(expected_commit):
        raise BronzeSourceError("Bronze source git_commit must be 40 lowercase hex chars")
    repository_root = Path(data_root).resolve().parent
    actual_commit = _git(repository_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise BronzeSourceError(
            "Bronze source commit mismatch: "
            f"expected {expected_commit}, got {actual_commit}"
        )
    changes = _git(repository_root, "status", "--porcelain")
    if changes:
        preview = ", ".join(changes.splitlines()[:5])
        raise BronzeSourceError(
            f"Bronze source contains local changes and is not immutable: {preview}"
        )
    return BronzeSourceReport(
        repository_root=repository_root,
        expected_commit=expected_commit,
        actual_commit=actual_commit,
        tracked_tree_clean=True,
    )


def _git(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BronzeSourceError("Git is required to validate the Bronze source") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown Git error").strip()
        raise BronzeSourceError(
            f"cannot validate Bronze repository {repository_root}: {detail}"
        ) from exc
    return result.stdout.strip()


__all__ = [
    "BronzeSourceError",
    "BronzeSourceReport",
    "validate_bronze_repository",
]
