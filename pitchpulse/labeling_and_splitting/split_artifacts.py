"""Memory-bounded chronological split artifacts for prepared VAEP labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pitchpulse.dataset.training_dataset_validator import (
    load_training_dataset_manifest,
    single_file_training_dataset,
)
from pitchpulse.shared.file_io import file_sha256, read_json, write_json
from .splits import (
    SPLIT_NAMES,
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
        dataset_manifest_path: Path | None = None,
        match_metadata_path: Path | None = None,
        progress: Any | None = None,
    ) -> SplitArtifactPaths:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "VAEP split generation requires pyarrow; run `uv sync`"
            ) from exc

        if (dataset_manifest_path is None) == (match_metadata_path is None):
            raise ValueError(
                "provide exactly one of dataset_manifest_path or match_metadata_path"
            )
        dataset = (
            load_training_dataset_manifest(dataset_manifest_path)
            if dataset_manifest_path is not None
            else single_file_training_dataset(match_metadata_path)
        )
        directory = Path(label_directory).resolve()
        label_manifest = read_json(directory / "label_manifest.json")
        labels_path = directory / "action_labels.parquet"
        expected_hash = (label_manifest.get("files") or {}).get(labels_path.name)
        if progress is not None:
            progress("verifying action labels hash")
        if not isinstance(expected_hash, str) or file_sha256(labels_path) != expected_hash:
            raise ValueError("action_labels.parquet hash does not match label manifest")
        if label_manifest.get("target_policy_version") != TARGET_POLICY_VERSION:
            raise ValueError("label artifact uses an unexpected target policy")

        match_ids = sorted(dataset.match_ids)
        action_stub = pa.table(
            {
                "match_id": pa.array(match_ids, type=pa.int64()),
                "action_id": pa.array([0] * len(match_ids), type=pa.int64()),
            }
        )
        split_dataset = ChronologicalMatchSplitter().build(
            action_stub,
            dataset.matches,
            dataset_fingerprint=str(
                label_manifest["source_dataset_fingerprint"]
            ),
            target_policy_version=TARGET_POLICY_VERSION,
            match_metadata_sha256=dataset.metadata_fingerprint,
        )
        assignments = split_dataset.assignments
        manifest = split_dataset.manifest
        records = assignments.to_pylist()
        split_by_match = {
            int(row["match_id"]): str(row["split"]) for row in records
        }
        summaries: dict[str, dict[str, Any]] = {}
        for split in SPLIT_NAMES:
            summary = dict(manifest["splits"][split])
            summary.update(
                {
                    "action_count": 0,
                    "eligible_label_count": 0,
                    "scores_positive_count": 0,
                    "concedes_positive_count": 0,
                }
            )
            summaries[split] = summary

        parquet_file = pq.ParquetFile(labels_path)
        seen_match_ids: set[int] = set()
        action_count = 0
        try:
            for row_group in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(
                    row_group,
                    columns=("match_id", "scores", "concedes", "eligible"),
                )
                frame = table.to_pandas()
                frame["split"] = frame["match_id"].map(split_by_match)
                if frame["split"].isna().any():
                    raise ValueError("labels contain matches outside the selected dataset")
                seen_match_ids.update(
                    int(value) for value in frame["match_id"].unique()
                )
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
        finally:
            parquet_file.close()

        if seen_match_ids != set(match_ids):
            raise ValueError("label match IDs do not exactly match the dataset")
        if action_count != int(label_manifest["action_count"]):
            raise ValueError("split action count does not match the label manifest")

        for split in SPLIT_NAMES:
            summary = summaries[split]
            eligible_count = int(summary["eligible_label_count"])
            for target in ("scores", "concedes"):
                positive = int(summary[f"{target}_positive_count"])
                summary[f"{target}_positive_rate"] = (
                    round(positive / eligible_count, 12)
                    if eligible_count
                    else None
                )

        manifest["source_label_manifest"] = str(directory / "label_manifest.json")
        manifest["source_action_labels_sha256"] = expected_hash
        manifest["counts"]["action_count"] = action_count
        manifest["splits"] = summaries
        manifest["checks"]["all_action_matches_assigned"] = (
            seen_match_ids == set(split_by_match)
        )
        paths = SplitArtifactPaths(
            directory=directory,
            assignments=directory / "split_assignments.parquet",
            manifest=directory / "split_manifest.json",
        )
        self._write_parquet(pq, assignments, paths.assignments)
        manifest["files"] = {
            paths.assignments.name: file_sha256(paths.assignments),
        }
        write_json(paths.manifest, manifest)
        if progress is not None:
            progress(f"completed split artifact: {directory}")
        return paths

    @staticmethod
    def _write_parquet(pq: Any, table: Any, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)

__all__ = ["ChunkedSplitArtifactWriter", "SplitArtifactPaths"]
