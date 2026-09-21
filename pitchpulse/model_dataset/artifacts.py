"""Memory-bounded model-dataset join for the full VAEP corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pitchpulse.vaep_features.feature_builder import BASE_FEATURE_VERSION
from pitchpulse.shared.file_io import file_sha256, read_json, write_json

from pitchpulse.vaep_features.feature_dataset_loader import baseline_feature_allowlist
from .feature_allowlist import (
    FEATURE_ALLOWLIST_FILENAME,
    MODEL_DATASET_FILENAME,
    MODEL_DATASET_VERSION,
    TRACE_COLUMNS,
    VERSION_COLUMNS,
    feature_allowlist_manifest,
    validate_feature_allowlist,
)
from pitchpulse.labeling_and_splitting.splits import SPLIT_NAMES, SPLIT_VERSION
from pitchpulse.labeling_and_splitting.targets import (
    LABEL_COLUMNS,
    TARGET_POLICY_VERSION,
)


MODEL_DATASET_MANIFEST_FILENAME = "model_dataset_manifest.json"
LABEL_SCHEMA = (
    "analytics_run_id",
    "match_id",
    "action_id",
    "scores",
    "concedes",
    "eligible",
    "exclusion_reason",
    "target_policy_version",
)
SPLIT_SCHEMA = (
    "match_id",
    "competition_id",
    "season_id",
    "match_date",
    "split",
    "split_order",
    "split_version",
)


@dataclass(frozen=True, slots=True)
class ChunkedModelDatasetPaths:
    directory: Path
    model_dataset: Path
    feature_allowlist: Path
    manifest: Path


class ChunkedModelDatasetWriter:
    """Join aligned feature artifact features and labels one row group at a time."""

    def write(
        self,
        feature_artifact_directory: Path,
        label_directory: Path,
        *,
        progress: Any | None = None,
    ) -> ChunkedModelDatasetPaths:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "model dataset construction requires pyarrow; install requirements.txt"
            ) from exc

        # 1. Locate the three source artifacts and verify their manifests.
        feature_artifact_directory = Path(feature_artifact_directory).resolve()
        label_directory = Path(label_directory).resolve()
        feature_manifest, label_manifest, split_manifest, fingerprint = (
            self._load_manifests(feature_artifact_directory, label_directory)
        )

        features_path = feature_artifact_directory / "action_features.parquet"
        labels_path = label_directory / "action_labels.parquet"
        splits_path = label_directory / "split_assignments.parquet"
        self._verify_source_files(
            features_path=features_path,
            labels_path=labels_path,
            splits_path=splits_path,
            feature_manifest=feature_manifest,
            label_manifest=label_manifest,
            split_manifest=split_manifest,
            progress=progress,
        )

        allowlist = validate_feature_allowlist(baseline_feature_allowlist())
        paths = ChunkedModelDatasetPaths(
            directory=label_directory,
            model_dataset=label_directory / MODEL_DATASET_FILENAME,
            feature_allowlist=label_directory / FEATURE_ALLOWLIST_FILENAME,
            manifest=label_directory / MODEL_DATASET_MANIFEST_FILENAME,
        )

        # 2. Validate table layouts, then join and write one row group at a time.
        feature_file = None
        label_file = None
        try:
            feature_file = pq.ParquetFile(features_path)
            label_file = pq.ParquetFile(labels_path)
            split_by_match = self._validate_inputs(
                feature_file,
                label_file,
                pq.read_table(splits_path),
                allowlist,
            )
            eligible_count, seen_match_ids = self._write_batches(
                pa=pa,
                pq=pq,
                feature_file=feature_file,
                label_file=label_file,
                split_by_match=split_by_match,
                split_manifest=split_manifest,
                allowlist=allowlist,
                fingerprint=fingerprint,
                output_path=paths.model_dataset,
                progress=progress,
            )

            # 3. Write the small metadata files only after the dataset is complete.
            self._write_metadata(
                paths=paths,
                allowlist=allowlist,
                fingerprint=fingerprint,
                eligible_count=eligible_count,
                match_count=len(seen_match_ids),
            )
            if progress is not None:
                progress(f"completed model dataset: {paths.model_dataset}")
            return paths
        finally:
            if feature_file is not None:
                feature_file.close()
            if label_file is not None:
                label_file.close()

    @staticmethod
    def _load_manifests(
        feature_directory: Path,
        label_directory: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        feature_manifest = read_json(feature_directory / "manifest.json")
        label_manifest = read_json(label_directory / "label_manifest.json")
        split_manifest = read_json(label_directory / "split_manifest.json")
        fingerprint = str(feature_manifest.get("dataset_fingerprint"))
        if label_manifest.get("source_dataset_fingerprint") != fingerprint:
            raise ValueError("labels do not belong to the selected feature artifact")
        if split_manifest.get("source_dataset_fingerprint") != fingerprint:
            raise ValueError("splits do not belong to the selected feature artifact")
        return feature_manifest, label_manifest, split_manifest, fingerprint

    def _verify_source_files(
        self,
        *,
        features_path: Path,
        labels_path: Path,
        splits_path: Path,
        feature_manifest: dict[str, Any],
        label_manifest: dict[str, Any],
        split_manifest: dict[str, Any],
        progress: Any | None,
    ) -> None:
        sources = (
            (features_path, feature_manifest, "feature artifact features"),
            (labels_path, label_manifest, "action labels"),
            (splits_path, split_manifest, "split assignments"),
        )
        for path, manifest, label in sources:
            self._verify_declared_hash(path, manifest, progress, label)

    @staticmethod
    def _validate_inputs(
        feature_file: Any,
        label_file: Any,
        assignments: Any,
        allowlist: tuple[str, ...],
    ) -> dict[int, str]:
        expected_feature_columns = (
            "match_id",
            "action_id",
            "feature_version",
            *allowlist,
        )
        if tuple(feature_file.schema_arrow.names) != expected_feature_columns:
            raise ValueError("feature artifact schema is not the baseline allowlist")
        if feature_file.num_row_groups != label_file.num_row_groups:
            raise ValueError("feature and label row-group counts do not reconcile")
        if tuple(label_file.schema_arrow.names) != LABEL_SCHEMA:
            raise ValueError("action labels have an unexpected column layout")
        if tuple(assignments.column_names) != SPLIT_SCHEMA:
            raise ValueError("split assignments have an unexpected column layout")

        assignment_rows = assignments.to_pylist()
        split_by_match = {
            int(row["match_id"]): str(row["split"]) for row in assignment_rows
        }
        if len(split_by_match) != len(assignment_rows):
            raise ValueError("split assignments contain duplicate match_id values")
        if set(split_by_match.values()) != set(SPLIT_NAMES):
            raise ValueError("split assignments contain invalid split names")
        if any(row["split_version"] != SPLIT_VERSION for row in assignment_rows):
            raise ValueError("split assignments use an unexpected version")
        return split_by_match

    def _write_batches(
        self,
        *,
        pa: Any,
        pq: Any,
        feature_file: Any,
        label_file: Any,
        split_by_match: dict[int, str],
        split_manifest: dict[str, Any],
        allowlist: tuple[str, ...],
        fingerprint: str,
        output_path: Path,
        progress: Any | None,
    ) -> tuple[int, set[int]]:
        temporary = output_path.with_suffix(".parquet.tmp")
        temporary.unlink(missing_ok=True)
        writer = None
        eligible_count = 0
        seen_match_ids: set[int] = set()
        metadata = self._dataset_metadata(fingerprint, allowlist)

        try:
            for row_group in range(feature_file.num_row_groups):
                table, match_ids = self._build_batch(
                    pa,
                    feature_file.read_row_group(row_group),
                    label_file.read_row_group(row_group),
                    split_by_match,
                    allowlist,
                    metadata,
                    batch_number=row_group + 1,
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                eligible_count += table.num_rows
                seen_match_ids.update(match_ids)
                if progress is not None:
                    progress(
                        f"joined batch {row_group + 1}/{feature_file.num_row_groups}: "
                        f"{table.num_rows} eligible rows"
                    )

            if writer is None:
                raise ValueError("feature artifact contains no row groups")
            writer.close()
            writer = None
            self._verify_output_counts(
                eligible_count,
                seen_match_ids,
                split_by_match,
                split_manifest,
            )
            temporary.replace(output_path)
            return eligible_count, seen_match_ids
        except BaseException:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _build_batch(
        pa: Any,
        features: Any,
        labels: Any,
        split_by_match: dict[int, str],
        allowlist: tuple[str, ...],
        metadata: dict[bytes, bytes],
        *,
        batch_number: int,
    ) -> tuple[Any, set[int]]:
        if features.num_rows != labels.num_rows:
            raise ValueError(
                f"feature/label row count mismatch in batch {batch_number}"
            )
        for key in ("match_id", "action_id"):
            if not features[key].equals(labels[key]):
                raise ValueError(
                    f"feature/label {key} mismatch in batch {batch_number}"
                )

        selected_features = features.filter(labels["eligible"])
        selected_labels = labels.filter(labels["eligible"])
        if any(
            value != TARGET_POLICY_VERSION
            for value in selected_labels["target_policy_version"].to_pylist()
        ):
            raise ValueError("labels use an unexpected target policy version")
        if (
            selected_labels["scores"].null_count
            or selected_labels["concedes"].null_count
        ):
            raise ValueError("eligible labels cannot contain null targets")

        match_ids = {int(value) for value in selected_features["match_id"].to_pylist()}
        try:
            split_values = [
                split_by_match[int(value)]
                for value in selected_features["match_id"].to_pylist()
            ]
        except KeyError as exc:
            raise ValueError(
                f"feature match has no split assignment: {exc.args[0]}"
            ) from exc

        arrays = [
            selected_labels["analytics_run_id"],
            selected_features["match_id"],
            selected_features["action_id"],
        ]
        names = list(TRACE_COLUMNS)
        arrays.extend(selected_features[name] for name in allowlist)
        names.extend(allowlist)
        arrays.extend(
            [
                selected_labels["scores"],
                selected_labels["concedes"],
                pa.array(split_values, type=pa.string()),
                selected_features["feature_version"],
                selected_labels["target_policy_version"],
                pa.array(
                    [SPLIT_VERSION] * selected_features.num_rows,
                    type=pa.string(),
                ),
            ]
        )
        names.extend(LABEL_COLUMNS + ("split",) + VERSION_COLUMNS)
        table = pa.Table.from_arrays(arrays, names=names)
        return table.replace_schema_metadata(metadata), match_ids

    @staticmethod
    def _verify_output_counts(
        eligible_count: int,
        seen_match_ids: set[int],
        split_by_match: dict[int, str],
        split_manifest: dict[str, Any],
    ) -> None:
        expected_eligible = sum(
            int(value["eligible_label_count"])
            for value in split_manifest["splits"].values()
        )
        if eligible_count != expected_eligible:
            raise ValueError("model dataset eligible-row count does not reconcile")
        if seen_match_ids != set(split_by_match):
            raise ValueError("model dataset match IDs do not reconcile with splits")

    @staticmethod
    def _dataset_metadata(
        fingerprint: str, allowlist: tuple[str, ...]
    ) -> dict[bytes, bytes]:
        return {
            b"model_dataset_version": MODEL_DATASET_VERSION.encode(),
            b"source_dataset_fingerprint": fingerprint.encode(),
            b"feature_version": BASE_FEATURE_VERSION.encode(),
            b"target_policy_version": TARGET_POLICY_VERSION.encode(),
            b"split_version": SPLIT_VERSION.encode(),
            b"feature_allowlist_sha256": ChunkedModelDatasetWriter._canonical_sha256(
                list(allowlist)
            ).encode(),
        }

    @staticmethod
    def _write_metadata(
        *,
        paths: ChunkedModelDatasetPaths,
        allowlist: tuple[str, ...],
        fingerprint: str,
        eligible_count: int,
        match_count: int,
    ) -> None:
        allowlist_document = feature_allowlist_manifest(
            allowlist,
            source_dataset_fingerprint=fingerprint,
            feature_version=BASE_FEATURE_VERSION,
        )
        write_json(paths.feature_allowlist, allowlist_document)
        manifest = {
            "schema_version": 1,
            "model_dataset_version": MODEL_DATASET_VERSION,
            "source_dataset_fingerprint": fingerprint,
            "target_policy_version": TARGET_POLICY_VERSION,
            "split_version": SPLIT_VERSION,
            "row_count": eligible_count,
            "match_count": match_count,
            "feature_count": len(allowlist),
            "files": {
                paths.model_dataset.name: file_sha256(paths.model_dataset),
                paths.feature_allowlist.name: file_sha256(paths.feature_allowlist),
            },
        }
        write_json(paths.manifest, manifest)

    @staticmethod
    def _verify_declared_hash(
        path: Path, manifest: dict[str, Any], progress: Any, label: str
    ) -> None:
        expected = (manifest.get("files") or {}).get(path.name)
        if progress is not None:
            progress(f"verifying {label} hash")
        if not isinstance(expected, str) or file_sha256(path) != expected:
            raise ValueError(f"{label} hash does not match its manifest")

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True) + "\n"
        return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "MODEL_DATASET_MANIFEST_FILENAME",
    "ChunkedModelDatasetPaths",
    "ChunkedModelDatasetWriter",
]
