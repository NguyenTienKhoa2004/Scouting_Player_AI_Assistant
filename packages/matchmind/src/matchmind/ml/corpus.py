"""Pinned multi-competition corpus and production-adequacy checks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .splits import MatchMetadata, SPLIT_NAMES, load_statsbomb_match_metadata


TRAINING_CORPUS_SCHEMA_VERSION = 1
DEFAULT_ADEQUACY_POLICY: dict[str, int | bool] = {
    "minimum_matches": 1500,
    "minimum_competitions": 4,
    "minimum_competition_seasons": 6,
    "minimum_matches_per_split": 75,
    "minimum_positive_labels_per_target_per_split": 250,
    "require_complete_declared_corpus": True,
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TrainingCorpusError(ValueError):
    """Raised when a declared training corpus is invalid or inconsistent."""


@dataclass(frozen=True, slots=True)
class CorpusSelection:
    competition_id: int
    competition_name: str
    season_id: int
    season_name: str
    matches_path: Path
    expected_matches: int
    sha256: str
    matches: tuple[MatchMetadata, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "competition_id": self.competition_id,
            "competition_name": self.competition_name,
            "season_id": self.season_id,
            "season_name": self.season_name,
            "matches_path": str(self.matches_path),
            "expected_matches": self.expected_matches,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TrainingCorpus:
    corpus_id: str
    manifest_path: Path | None
    manifest_sha256: str | None
    source: dict[str, Any]
    scope: dict[str, Any]
    adequacy_policy: dict[str, int | bool]
    selections: tuple[CorpusSelection, ...]
    matches: tuple[MatchMetadata, ...]
    metadata_fingerprint: str

    @property
    def match_ids(self) -> frozenset[int]:
        return frozenset(match.match_id for match in self.matches)

    def assess(
        self,
        materialized_match_ids: Iterable[int],
        split_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Assess the materialized feature artifact rows, never only the declaration."""

        materialized_ids = frozenset(int(value) for value in materialized_match_ids)
        unexpected = sorted(materialized_ids - self.match_ids)
        if unexpected:
            raise TrainingCorpusError(
                f"feature artifact contains matches outside the corpus: {unexpected}"
            )

        selected = tuple(
            match for match in self.matches if match.match_id in materialized_ids
        )
        competitions = {match.competition_id for match in selected}
        competition_seasons = {
            (match.competition_id, match.season_id) for match in selected
        }
        policy = self.adequacy_policy
        checks: dict[str, bool] = {
            "minimum_matches": len(selected) >= int(policy["minimum_matches"]),
            "minimum_competitions": len(competitions)
            >= int(policy["minimum_competitions"]),
            "minimum_competition_seasons": len(competition_seasons)
            >= int(policy["minimum_competition_seasons"]),
            "complete_declared_corpus": (
                materialized_ids == self.match_ids
                if bool(policy["require_complete_declared_corpus"])
                else True
            ),
        }

        split_checks: dict[str, dict[str, bool]] = {}
        split_summaries = split_manifest.get("splits")
        if not isinstance(split_summaries, dict):
            raise TrainingCorpusError("split manifest has no splits object")
        for split in SPLIT_NAMES:
            summary = split_summaries.get(split)
            if not isinstance(summary, dict):
                raise TrainingCorpusError(f"split manifest is missing {split}")
            item = {
                "minimum_matches": int(summary.get("match_count", 0))
                >= int(policy["minimum_matches_per_split"]),
                "minimum_scores_positive": int(
                    summary.get("scores_positive_count", 0)
                )
                >= int(policy["minimum_positive_labels_per_target_per_split"]),
                "minimum_concedes_positive": int(
                    summary.get("concedes_positive_count", 0)
                )
                >= int(policy["minimum_positive_labels_per_target_per_split"]),
            }
            split_checks[split] = item

        blockers = [name for name, passed in checks.items() if not passed]
        blockers.extend(
            f"{split}.{name}"
            for split, values in split_checks.items()
            for name, passed in values.items()
            if not passed
        )
        declared_competitions = {match.competition_id for match in self.matches}
        declared_competition_seasons = {
            (match.competition_id, match.season_id) for match in self.matches
        }
        return {
            "corpus_id": self.corpus_id,
            "source": dict(self.source),
            "scope": dict(self.scope),
            "source_manifest_path": (
                str(self.manifest_path) if self.manifest_path is not None else None
            ),
            "source_manifest_sha256": self.manifest_sha256,
            "metadata_fingerprint": self.metadata_fingerprint,
            "adequacy_policy": dict(policy),
            "declared": {
                "match_count": len(self.matches),
                "competition_count": len(declared_competitions),
                "competition_season_count": len(declared_competition_seasons),
                "selections": [selection.as_dict() for selection in self.selections],
            },
            "materialized": {
                "match_count": len(selected),
                "competition_count": len(competitions),
                "competition_season_count": len(competition_seasons),
                "coverage_rate": round(len(selected) / len(self.matches), 12),
            },
            "checks": checks,
            "split_checks": split_checks,
            "adequate_for_production_evaluation": not blockers,
            "blockers": blockers,
        }


def load_training_corpus_manifest(path: Path) -> TrainingCorpus:
    """Load, hash, and reconcile every StatsBomb competition-season selection."""

    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != TRAINING_CORPUS_SCHEMA_VERSION:
        raise TrainingCorpusError(
            f"training corpus schema_version must be {TRAINING_CORPUS_SCHEMA_VERSION}"
        )
    corpus_id = manifest.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise TrainingCorpusError("training corpus requires a non-empty corpus_id")

    policy = _validate_policy(manifest.get("adequacy_policy"))
    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise TrainingCorpusError("training corpus scope must be an object")
    raw_selections = manifest.get("selections")
    if not isinstance(raw_selections, list) or not raw_selections:
        raise TrainingCorpusError("training corpus selections must be non-empty")

    selections: list[CorpusSelection] = []
    all_matches: list[MatchMetadata] = []
    seen_pairs: set[tuple[int, int]] = set()
    seen_match_ids: set[int] = set()
    for index, value in enumerate(raw_selections):
        if not isinstance(value, dict):
            raise TrainingCorpusError(f"selection {index} must be an object")
        competition_id = _positive_int(value.get("competition_id"), "competition_id")
        season_id = _positive_int(value.get("season_id"), "season_id")
        pair = (competition_id, season_id)
        if pair in seen_pairs:
            raise TrainingCorpusError(f"duplicate competition-season selection: {pair}")
        seen_pairs.add(pair)
        expected_matches = _positive_int(
            value.get("expected_matches"), "expected_matches"
        )
        expected_hash = value.get("sha256")
        if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(
            expected_hash
        ):
            raise TrainingCorpusError(f"selection {pair} has invalid sha256")
        relative_path = value.get("matches_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise TrainingCorpusError(f"selection {pair} requires matches_path")
        metadata_path = (manifest_path.parent / relative_path).resolve()
        actual_hash = _sha256(metadata_path)
        if actual_hash != expected_hash:
            raise TrainingCorpusError(
                f"SHA-256 mismatch for {metadata_path}: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        matches = load_statsbomb_match_metadata(metadata_path)
        if len(matches) != expected_matches:
            raise TrainingCorpusError(
                f"selection {pair} expected {expected_matches} matches, got "
                f"{len(matches)}"
            )
        if any(
            match.competition_id != competition_id or match.season_id != season_id
            for match in matches
        ):
            raise TrainingCorpusError(
                f"selection {pair} contains different competition/season metadata"
            )
        if scope.get("event_and_lineup_files_required") is True:
            data_root = metadata_path.parents[2]
            missing_files = [
                str(path)
                for match in matches
                for path in (
                    data_root / "events" / f"{match.match_id}.json",
                    data_root / "lineups" / f"{match.match_id}.json",
                )
                if not path.is_file()
            ]
            if missing_files:
                raise TrainingCorpusError(
                    f"selection {pair} is missing required source files: "
                    f"{missing_files[:10]}"
                )
        duplicates = seen_match_ids & {match.match_id for match in matches}
        if duplicates:
            raise TrainingCorpusError(
                f"match IDs occur in multiple selections: {sorted(duplicates)}"
            )
        seen_match_ids.update(match.match_id for match in matches)
        selection = CorpusSelection(
            competition_id=competition_id,
            competition_name=_nonempty_text(
                value.get("competition_name"), "competition_name"
            ),
            season_id=season_id,
            season_name=_nonempty_text(value.get("season_name"), "season_name"),
            matches_path=metadata_path,
            expected_matches=expected_matches,
            sha256=expected_hash,
            matches=matches,
        )
        selections.append(selection)
        all_matches.extend(matches)

    ordered_matches = tuple(
        sorted(all_matches, key=lambda match: match.chronological_key)
    )
    _validate_expected_summary(manifest.get("expected_summary"), ordered_matches)
    metadata_fingerprint = _canonical_sha256(
        [
            {
                "competition_id": selection.competition_id,
                "season_id": selection.season_id,
                "sha256": selection.sha256,
            }
            for selection in selections
        ]
    )
    return TrainingCorpus(
        corpus_id=corpus_id,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        source=dict(manifest.get("source") or {}),
        scope=dict(scope),
        adequacy_policy=policy,
        selections=tuple(selections),
        matches=ordered_matches,
        metadata_fingerprint=metadata_fingerprint,
    )


def single_file_training_corpus(path: Path) -> TrainingCorpus:
    """Wrap the legacy one-file input while applying the production gate."""

    metadata_path = Path(path).resolve()
    matches = load_statsbomb_match_metadata(metadata_path)
    if not matches:
        raise TrainingCorpusError("single-file training corpus contains no matches")
    pairs = {(match.competition_id, match.season_id) for match in matches}
    if len(pairs) != 1:
        raise TrainingCorpusError(
            "legacy match metadata must contain exactly one competition-season"
        )
    competition_id, season_id = next(iter(pairs))
    digest = _sha256(metadata_path)
    selection = CorpusSelection(
        competition_id=competition_id,
        competition_name=f"competition-{competition_id}",
        season_id=season_id,
        season_name=f"season-{season_id}",
        matches_path=metadata_path,
        expected_matches=len(matches),
        sha256=digest,
        matches=matches,
    )
    return TrainingCorpus(
        corpus_id=f"single-{competition_id}-{season_id}",
        manifest_path=None,
        manifest_sha256=None,
        source={"type": "legacy_single_matches_file"},
        scope={},
        adequacy_policy=dict(DEFAULT_ADEQUACY_POLICY),
        selections=(selection,),
        matches=tuple(sorted(matches, key=lambda match: match.chronological_key)),
        metadata_fingerprint=_canonical_sha256(
            [
                {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "sha256": digest,
                }
            ]
        ),
    )


def _validate_policy(value: Any) -> dict[str, int | bool]:
    if not isinstance(value, dict):
        raise TrainingCorpusError("adequacy_policy must be an object")
    if set(value) != set(DEFAULT_ADEQUACY_POLICY):
        raise TrainingCorpusError(
            "adequacy_policy must declare the complete versioned policy"
        )
    result: dict[str, int | bool] = {}
    for name, default in DEFAULT_ADEQUACY_POLICY.items():
        item = value[name]
        if isinstance(default, bool):
            if not isinstance(item, bool):
                raise TrainingCorpusError(f"adequacy policy {name} must be boolean")
        else:
            item = _positive_int(item, name)
        result[name] = item
    return result


def _validate_expected_summary(value: Any, matches: tuple[MatchMetadata, ...]) -> None:
    if not isinstance(value, dict):
        raise TrainingCorpusError("expected_summary must be an object")
    actual = {
        "match_count": len(matches),
        "competition_count": len({match.competition_id for match in matches}),
        "competition_season_count": len(
            {(match.competition_id, match.season_id) for match in matches}
        ),
        "first_match_date": matches[0].match_date.isoformat(),
        "last_match_date": matches[-1].match_date.isoformat(),
    }
    if value != actual:
        raise TrainingCorpusError(
            f"expected_summary does not reconcile: expected {value!r}, got {actual!r}"
        )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingCorpusError(f"{field} must be a positive integer")
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingCorpusError(f"{field} must be non-empty text")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingCorpusError(
            f"cannot read training corpus manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TrainingCorpusError("training corpus manifest must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrainingCorpusError(
            f"cannot hash training corpus input {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "DEFAULT_ADEQUACY_POLICY",
    "TRAINING_CORPUS_SCHEMA_VERSION",
    "CorpusSelection",
    "TrainingCorpus",
    "TrainingCorpusError",
    "load_training_corpus_manifest",
    "single_file_training_corpus",
]
