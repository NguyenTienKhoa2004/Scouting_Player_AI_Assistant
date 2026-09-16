"""Memory-bounded chronological split artifacts for prepared VAEP labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matchmind.corpus.training_dataset_validator import (
    load_training_corpus_manifest,
    single_file_training_corpus,
)
from .splits import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_NAMES,
    SPLIT_VERSION,
    ChronologicalMatchSplitter,
)
from .targets import TARGET_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class SplitArtifactPaths:
    directory: Path
    assignments: Path
    manifest: Path


class ChunkedSplitArtifactWriter:
    """Create match assignments and label balance without loading all labels."""

    def write(
        self,
        label_directory: Path,
        *,
        corpus_manifest_path: Path | None = None,
        match_metadata_path: Path | None = None,
        progress: Any | None = None,
    ) -> SplitArtifactPaths:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "VAEP split generation requires pyarrow; install requirements.txt"
            ) from exc

        if (corpus_manifest_path is None) == (match_metadata_path is None):
            raise ValueError(
                "provide exactly one of corpus_manifest_path or match_metadata_path"
            )
        corpus = (
            load_training_corpus_manifest(corpus_manifest_path)
            if corpus_manifest_path is not None
            else single_file_training_corpus(match_metadata_path)
        )
        directory = Path(label_directory).resolve()
        label_manifest = self._read_json(directory / "label_manifest.json")
        labels_path = directory / "action_labels.parquet"
        expected_hash = (label_manifest.get("files") or {}).get(labels_path.name)
        if progress is not None:
            progress("verifying action labels hash")
        if not isinstance(expected_hash, str) or self._sha256(labels_path) != expected_hash:
            raise ValueError("action_labels.parquet hash does not match label manifest")
        if label_manifest.get("target_policy_version") != TARGET_POLICY_VERSION:
            raise ValueError("label artifact uses an unexpected target policy")

        match_ids = sorted(corpus.match_ids)
        action_stub = pa.table(
            {
                "match_id": pa.array(match_ids, type=pa.int64()),
                "action_id": pa.array([0] * len(match_ids), type=pa.int64()),
            }
        )
        split_dataset = ChronologicalMatchSplitter().build(
            action_stub,
            corpus.matches,
            dataset_fingerprint=str(
                label_manifest["source_dataset_fingerprint"]
            ),
            target_policy_version=TARGET_POLICY_VERSION,
            match_metadata_sha256=corpus.metadata_fingerprint,
        )
        assignments = split_dataset.assignments
        records = assignments.to_pylist()
        split_by_match = {
            int(row["match_id"]): str(row["split"]) for row in records
        }
        metadata_by_id = {match.match_id: match for match in corpus.matches}
        summaries = {
            split: {
                "match_count": sum(
                    str(row["split"]) == split for row in records
                ),
                "action_count": 0,
                "eligible_label_count": 0,
                "scores_positive_count": 0,
                "concedes_positive_count": 0,
            }
            for split in SPLIT_NAMES
        }

        parquet_file = pq.ParquetFile(labels_path)
        seen_match_ids: set[int] = set()
        action_count = 0
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group,
                columns=("match_id", "scores", "concedes", "eligible"),
            )
            frame = table.to_pandas()
            frame["split"] = frame["match_id"].map(split_by_match)
            if frame["split"].isna().any():
                raise ValueError("labels contain matches outside the selected corpus")
            seen_match_ids.update(int(value) for value in frame["match_id"].unique())
            action_count += len(frame)
            for split in SPLIT_NAMES:
                selected = frame[frame["split"] == split]
                eligible = selected[selected["eligible"].astype(bool)]
                summary = summaries[split]
                summary["action_count"] += int(len(selected))
                summary["eligible_label_count"] += int(len(eligible))
                summary["scores_positive_count"] += int(
                    eligible["scores"].eq(True).sum()
                )
                summary["concedes_positive_count"] += int(
                    eligible["concedes"].eq(True).sum()
                )
            if progress is not None:
                progress(
                    f"counted label batch {row_group + 1}/"
                    f"{parquet_file.num_row_groups}"
                )

        if seen_match_ids != set(match_ids):
            raise ValueError("label match IDs do not exactly match the corpus")
        if action_count != int(label_manifest["action_count"]):
            raise ValueError("split action count does not match the label manifest")

        ordered = sorted(records, key=lambda row: int(row["split_order"]))
        for split in SPLIT_NAMES:
            selected = [row for row in ordered if row["split"] == split]
            first = metadata_by_id[int(selected[0]["match_id"])]
            last = metadata_by_id[int(selected[-1]["match_id"])]
            summary = summaries[split]
            eligible_count = int(summary["eligible_label_count"])
            summary["first_match_datetime"] = first.match_datetime.isoformat()
            summary["last_match_datetime"] = last.match_datetime.isoformat()
            for target in ("scores", "concedes"):
                positive = int(summary[f"{target}_positive_count"])
                summary[f"{target}_positive_rate"] = (
                    round(positive / eligible_count, 12)
                    if eligible_count
                    else None
                )

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
        manifest = {
            "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_version": SPLIT_VERSION,
            "target_policy_version": TARGET_POLICY_VERSION,
            "source_dataset_fingerprint": label_manifest[
                "source_dataset_fingerprint"
            ],
            "source_match_metadata_sha256": corpus.metadata_fingerprint,
            "source_label_manifest": str(directory / "label_manifest.json"),
            "source_action_labels_sha256": expected_hash,
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
                "boundary_algorithm": (
                    "floor cumulative 70% and 85%, with non-empty splits"
                ),
                "random_seed": None,
                "uses_vaep_fit": False,
            },
            "counts": {
                "match_count": len(records),
                "action_count": action_count,
            },
            "splits": summaries,
            "checks": {
                "every_match_assigned_once": len(split_by_match) == len(records),
                "all_action_matches_assigned": seen_match_ids == set(split_by_match),
                "split_names_valid": set(split_by_match.values()) == set(SPLIT_NAMES),
                "chronological_order": True,
                "match_level_isolation": True,
            },
        }
        paths = SplitArtifactPaths(
            directory=directory,
            assignments=directory / "split_assignments.parquet",
            manifest=directory / "split_manifest.json",
        )
        self._write_parquet(pq, assignments, paths.assignments)
        manifest["files"] = {
            paths.assignments.name: self._sha256(paths.assignments),
        }
        self._write_json(paths.manifest, manifest)
        if progress is not None:
            progress(f"completed split artifact: {directory}")
        return paths

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}")
        return value

    @staticmethod
    def _write_parquet(pq: Any, table: Any, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = ["ChunkedSplitArtifactWriter", "SplitArtifactPaths"]
