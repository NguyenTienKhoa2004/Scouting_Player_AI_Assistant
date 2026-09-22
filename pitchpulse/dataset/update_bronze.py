"""Move Bronze to an explicit StatsBomb commit and create a new dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.dataset.bronze import (  # noqa: E402
    BronzeSourceError,
    validate_bronze_repository,
)
from pitchpulse.dataset.match_metadata import (  # noqa: E402
    MatchMetadata,
    load_statsbomb_match_metadata,
)
from pitchpulse.dataset.training_dataset_validator import (  # noqa: E402
    load_training_dataset_manifest,
)
from pitchpulse.shared.paths import PROJECT_ROOT  # noqa: E402


DEFAULT_DATASET_MANIFEST = (
    PROJECT_ROOT / "configs" / "datasets" / "vaep-training-dataset-v1.json"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checkout one explicit StatsBomb commit and create a reconciled, "
            "versioned dataset manifest. There is intentionally no latest mode."
        )
    )
    parser.add_argument("--commit", required=True, help="Exact 40-character commit SHA.")
    parser.add_argument("--dataset-id", required=True, help="New versioned dataset ID.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Require the explicit commit to already exist in the local clone.",
    )
    return parser.parse_args()


def update_bronze_repository(
    *,
    source_manifest: Path,
    output_manifest: Path,
    commit: str,
    dataset_id: str,
    fetch: bool = True,
) -> Path:
    """Checkout ``commit`` and atomically create a validated manifest beside source."""

    if not _COMMIT_PATTERN.fullmatch(commit):
        raise BronzeSourceError("--commit must be an exact 40-character lowercase SHA")
    if not dataset_id.strip():
        raise BronzeSourceError("--dataset-id must be non-empty")

    source_path = Path(source_manifest).resolve()
    output_path = Path(output_manifest).resolve()
    if output_path.exists():
        raise BronzeSourceError(f"output manifest already exists: {output_path}")
    if output_path.parent != source_path.parent:
        raise BronzeSourceError(
            "output manifest must be beside the source manifest so relative Bronze "
            "paths remain stable"
        )

    original = _read_manifest(source_path)
    data_root = _resolve_data_root(source_path, original)
    repository_root = data_root.parent
    current_dataset = load_training_dataset_manifest(source_path)
    current_commit = current_dataset.source.get("git_commit")
    if not isinstance(current_commit, str):
        raise BronzeSourceError("source manifest does not contain a pinned commit")
    validate_bronze_repository(data_root, current_commit)
    old_commit = _run_git(repository_root, "rev-parse", "HEAD")
    changes = _run_git(repository_root, "status", "--porcelain")
    if changes:
        raise BronzeSourceError("Bronze repository has local changes; update aborted")

    if fetch:
        _run_git(repository_root, "fetch", "origin")
    _run_git(repository_root, "cat-file", "-e", f"{commit}^{{commit}}")

    output_created = False
    try:
        _run_git(repository_root, "checkout", "--detach", commit)
        validate_bronze_repository(data_root, commit)
        updated = build_updated_manifest(
            original,
            manifest_path=source_path,
            commit=commit,
            dataset_id=dataset_id,
        )
        with output_path.open("x", encoding="utf-8", newline="\n") as destination:
            output_created = True
            json.dump(updated, destination, indent=2)
            destination.write("\n")
        load_training_dataset_manifest(output_path)
    except BaseException:
        if output_created:
            output_path.unlink(missing_ok=True)
        _run_git(repository_root, "checkout", "--detach", old_commit)
        raise
    return output_path


def build_updated_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    commit: str,
    dataset_id: str,
) -> dict[str, Any]:
    """Reconcile declared selections and summary against the checked-out snapshot."""

    updated = json.loads(json.dumps(manifest))
    updated["dataset_id"] = dataset_id
    source = updated.get("source")
    if not isinstance(source, dict):
        raise BronzeSourceError("manifest source must be an object")
    source["git_commit"] = commit
    data_root = _resolve_data_root(manifest_path, updated)

    selections = updated.get("selections")
    if not isinstance(selections, list) or not selections:
        raise BronzeSourceError("manifest selections must be a non-empty array")
    all_matches: list[MatchMetadata] = []
    for selection in selections:
        if not isinstance(selection, dict):
            raise BronzeSourceError("each manifest selection must be an object")
        relative_path = selection.get("matches_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise BronzeSourceError("selection matches_path must be non-empty text")
        matches_path = (data_root / relative_path).resolve()
        try:
            matches_path.relative_to(data_root)
        except ValueError as exc:
            raise BronzeSourceError("selection matches_path escapes Bronze data") from exc
        matches = load_statsbomb_match_metadata(matches_path)
        if not matches:
            raise BronzeSourceError(f"selection has no matches: {matches_path}")
        pair = (selection.get("competition_id"), selection.get("season_id"))
        if any(
            (match.competition_id, match.season_id) != pair for match in matches
        ):
            raise BronzeSourceError(
                f"selection {pair} contains another competition or season"
            )
        selection["expected_matches"] = len(matches)
        selection["sha256"] = _file_sha256(matches_path)
        all_matches.extend(matches)

    ordered = sorted(all_matches, key=lambda match: match.chronological_key)
    updated["expected_summary"] = {
        "match_count": len(ordered),
        "competition_count": len({match.competition_id for match in ordered}),
        "competition_season_count": len(
            {(match.competition_id, match.season_id) for match in ordered}
        ),
        "first_match_date": ordered[0].match_date.isoformat(),
        "last_match_date": ordered[-1].match_date.isoformat(),
    }
    return updated


def _resolve_data_root(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    bronze = manifest.get("bronze")
    if not isinstance(bronze, dict):
        raise BronzeSourceError("manifest bronze must be an object")
    value = bronze.get("data_root")
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise BronzeSourceError("bronze.data_root must be a relative path")
    return (manifest_path.parent / value).resolve()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BronzeSourceError(f"cannot read source manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BronzeSourceError("source manifest must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BronzeSourceError("Git is required to update Bronze") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown Git error").strip()
        raise BronzeSourceError(f"Bronze Git update failed: {detail}") from exc
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    output = update_bronze_repository(
        source_manifest=args.source_manifest,
        output_manifest=args.output_manifest,
        commit=args.commit,
        dataset_id=args.dataset_id,
        fetch=not args.no_fetch,
    )
    print(f"Bronze updated to explicit commit {args.commit}")
    print(f"Validated manifest written to {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Bronze update failed: {exc}") from exc
