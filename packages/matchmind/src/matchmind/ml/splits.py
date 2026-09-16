"""Deterministic chronological match-level dataset splitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable


SPLIT_VERSION = "chronological-match-70-15-15-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")


class SplitAssignmentError(ValueError):
    """Raised when matches cannot be assigned without leakage."""


@dataclass(frozen=True, slots=True)
class MatchMetadata:
    match_id: int
    competition_id: int
    season_id: int
    match_date: date
    kick_off: time

    @property
    def chronological_key(self) -> tuple[date, time, int, int, int]:
        return (
            self.match_date,
            self.kick_off,
            self.competition_id,
            self.season_id,
            self.match_id,
        )

    @property
    def match_datetime(self) -> datetime:
        return datetime.combine(self.match_date, self.kick_off)


@dataclass(frozen=True, slots=True)
class SplitDataset:
    assignments: Any
    manifest: dict[str, Any]


def load_statsbomb_match_metadata(path: Path) -> tuple[MatchMetadata, ...]:
    """Read the minimal chronological contract from a StatsBomb matches file."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitAssignmentError(f"cannot read match metadata {source}: {exc}") from exc
    if not isinstance(value, list):
        raise SplitAssignmentError("StatsBomb matches metadata must be a JSON array")

    matches: list[MatchMetadata] = []
    for row in value:
        if not isinstance(row, dict):
            raise SplitAssignmentError("each match metadata row must be an object")
        competition = row.get("competition")
        season = row.get("season")
        if not isinstance(competition, dict) or not isinstance(season, dict):
            raise SplitAssignmentError("match metadata lacks competition or season")
        try:
            kick_off_text = str(row.get("kick_off") or "00:00:00")
            matches.append(
                MatchMetadata(
                    match_id=int(row["match_id"]),
                    competition_id=int(competition["competition_id"]),
                    season_id=int(season["season_id"]),
                    match_date=date.fromisoformat(str(row["match_date"])),
                    kick_off=time.fromisoformat(kick_off_text),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitAssignmentError(f"invalid match metadata row: {exc}") from exc

    ids = [match.match_id for match in matches]
    if len(ids) != len(set(ids)):
        raise SplitAssignmentError("match metadata contains duplicate match_id values")
    return tuple(matches)


class ChronologicalMatchSplitter:
    """Assign all actions from a match to one stable chronological split."""

    def build(
        self,
        actions: Any,
        matches: Iterable[MatchMetadata],
        *,
        dataset_fingerprint: str,
        target_policy_version: str,
        labels: Any | None = None,
        match_metadata_sha256: str | None = None,
    ) -> SplitDataset:
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError(
                "match splitting requires pyarrow; install requirements.txt"
            ) from exc

        if not {"match_id", "action_id"}.issubset(actions.column_names):
            raise SplitAssignmentError("actions must contain match_id and action_id")
        action_frame = actions.select(["match_id", "action_id"]).to_pandas()
        action_match_ids = {int(value) for value in action_frame["match_id"].unique()}
        metadata_by_id = {match.match_id: match for match in matches}
        missing = action_match_ids - set(metadata_by_id)
        if missing:
            raise SplitAssignmentError(
                f"match metadata is missing action matches: {sorted(missing)}"
            )
        ordered = sorted(
            (metadata_by_id[match_id] for match_id in action_match_ids),
            key=lambda match: match.chronological_key,
        )
        if len(ordered) < 3:
            raise SplitAssignmentError(
                "at least three matches are required for train/validation/test"
            )

        train_end = max(1, min(len(ordered) - 2, len(ordered) * 70 // 100))
        validation_end = max(
            train_end + 1,
            min(len(ordered) - 1, len(ordered) * 85 // 100),
        )
        records: list[dict[str, Any]] = []
        for split_order, match in enumerate(ordered):
            if split_order < train_end:
                split = "train"
            elif split_order < validation_end:
                split = "validation"
            else:
                split = "test"
            records.append(
                {
                    "match_id": match.match_id,
                    "competition_id": match.competition_id,
                    "season_id": match.season_id,
                    "match_date": match.match_date,
                    "split": split,
                    "split_order": split_order,
                    "split_version": SPLIT_VERSION,
                }
            )

        schema = pa.schema(
            [
                pa.field("match_id", pa.int64(), nullable=False),
                pa.field("competition_id", pa.int64(), nullable=False),
                pa.field("season_id", pa.int64(), nullable=False),
                pa.field("match_date", pa.date32(), nullable=False),
                pa.field("split", pa.string(), nullable=False),
                pa.field("split_order", pa.int64(), nullable=False),
                pa.field("split_version", pa.string(), nullable=False),
            ],
            metadata={b"split_version": SPLIT_VERSION.encode()},
        )
        assignments = pa.Table.from_pylist(records, schema=schema)
        manifest = self._manifest(
            records,
            ordered,
            action_frame,
            dataset_fingerprint=dataset_fingerprint,
            target_policy_version=target_policy_version,
            labels=labels,
            match_metadata_sha256=match_metadata_sha256,
        )
        return SplitDataset(assignments=assignments, manifest=manifest)

    @staticmethod
    def _manifest(
        records: list[dict[str, Any]],
        ordered: list[MatchMetadata],
        action_frame: Any,
        *,
        dataset_fingerprint: str,
        target_policy_version: str,
        labels: Any | None,
        match_metadata_sha256: str | None,
    ) -> dict[str, Any]:
        split_by_match = {row["match_id"]: row["split"] for row in records}
        action_frame = action_frame.copy()
        action_frame["split"] = action_frame["match_id"].map(split_by_match)
        label_frame = None
        if labels is not None:
            label_frame = labels.to_pandas().copy()
            label_frame["split"] = label_frame["match_id"].map(split_by_match)

        split_summary: dict[str, Any] = {}
        for name in SPLIT_NAMES:
            selected = [
                match
                for match, row in zip(ordered, records, strict=True)
                if row["split"] == name
            ]
            split_actions = action_frame[action_frame["split"] == name]
            item: dict[str, Any] = {
                "match_count": len(selected),
                "action_count": int(len(split_actions)),
                "first_match_datetime": selected[0].match_datetime.isoformat(),
                "last_match_datetime": selected[-1].match_datetime.isoformat(),
            }
            if label_frame is not None:
                split_labels = label_frame[
                    (label_frame["split"] == name) & label_frame["eligible"]
                ]
                item["eligible_label_count"] = int(len(split_labels))
                for target in ("scores", "concedes"):
                    positive = int(split_labels[target].eq(True).sum())
                    item[f"{target}_positive_count"] = positive
                    item[f"{target}_positive_rate"] = (
                        round(positive / len(split_labels), 12)
                        if len(split_labels)
                        else None
                    )
            split_summary[name] = item

        assignment_payload = [
            {
                **row,
                "match_date": row["match_date"].isoformat(),
            }
            for row in records
        ]
        assignment_fingerprint = hashlib.sha256(
            (
                json.dumps(
                    assignment_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        ).hexdigest()
        return {
            "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_version": SPLIT_VERSION,
            "target_policy_version": target_policy_version,
            "source_dataset_fingerprint": dataset_fingerprint,
            "source_match_metadata_sha256": match_metadata_sha256,
            "assignment_fingerprint": assignment_fingerprint,
            "policy": {
                "unit": "complete_match",
                "ordering": [
                    "match_date",
                    "kick_off",
                    "competition_id",
                    "season_id",
                    "match_id",
                ],
                "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
                "boundary_algorithm": "floor cumulative 70% and 85%, with non-empty splits",
                "random_seed": None,
                "uses_vaep_fit": False,
            },
            "counts": {
                "match_count": len(records),
                "action_count": int(len(action_frame)),
            },
            "splits": split_summary,
            "checks": {
                "every_match_assigned_once": len(split_by_match) == len(records),
                "all_action_matches_assigned": not action_frame["split"].isna().any(),
                "split_names_valid": set(split_by_match.values()) == set(SPLIT_NAMES),
                "chronological_order": all(
                    left.chronological_key <= right.chronological_key
                    for left, right in zip(ordered, ordered[1:])
                ),
                "match_level_isolation": True,
            },
        }


def split_context(assignments: Any) -> dict[int, dict[str, Any]]:
    """Convert persisted assignments to target-audit match context."""

    return {
        int(row["match_id"]): {
            "competition_id": int(row["competition_id"]),
            "season_id": int(row["season_id"]),
            "split": str(row["split"]),
        }
        for row in assignments.to_pylist()
    }


__all__ = [
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "SPLIT_NAMES",
    "SPLIT_VERSION",
    "ChronologicalMatchSplitter",
    "MatchMetadata",
    "SplitAssignmentError",
    "SplitDataset",
    "load_statsbomb_match_metadata",
    "split_context",
]
