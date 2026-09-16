"""Memory-bounded model-dataset join for the full VAEP corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matchmind.analytics.features.vaep_features import BASE_FEATURE_VERSION

from .feature_artifacts import baseline_feature_allowlist
from .dataset import (
    FEATURE_ALLOWLIST_FILENAME,
    MODEL_DATASET_FILENAME,
    MODEL_DATASET_VERSION,
    TRACE_COLUMNS,
    VERSION_COLUMNS,
    feature_allowlist_manifest,
    validate_feature_allowlist,
    ModelDatasetBuilder,
)
from .splits import SPLIT_NAMES, SPLIT_VERSION
from .targets import LABEL_COLUMNS, TARGET_POLICY_VERSION


MODEL_DATASET_MANIFEST_FILENAME = "model_dataset_manifest.json"


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

        feature_artifact_directory = Path(feature_artifact_directory).resolve()
        directory = Path(label_directory).resolve()
        feature_manifest = self._read_json(feature_artifact_directory / "manifest.json")
        label_manifest = self._read_json(directory / "label_manifest.json")
        split_manifest = self._read_json(directory / "split_manifest.json")
        fingerprint = str(feature_manifest.get("dataset_fingerprint"))
        if label_manifest.get("source_dataset_fingerprint") != fingerprint:
            raise ValueError("labels do not belong to the selected feature artifact artifact")
        if split_manifest.get("source_dataset_fingerprint") != fingerprint:
            raise ValueError("splits do not belong to the selected feature artifact artifact")

        features_path = feature_artifact_directory / "action_features.parquet"
        labels_path = directory / "action_labels.parquet"
        splits_path = directory / "split_assignments.parquet"
        self._verify_declared_hash(
            features_path, feature_manifest, progress, "feature artifact features"
        )
        self._verify_declared_hash(
            labels_path, label_manifest, progress, "action labels"
        )
        self._verify_declared_hash(
            splits_path, split_manifest, progress, "split assignments"
        )

        allowlist = validate_feature_allowlist(baseline_feature_allowlist())
        expected_feature_columns = (
            "match_id",
            "action_id",
            "feature_version",
            *allowlist,
        )
        feature_file = pq.ParquetFile(features_path)
        label_file = pq.ParquetFile(labels_path)
        if tuple(feature_file.schema_arrow.names) != expected_feature_columns:
            raise ValueError("feature artifact feature schema is not the baseline allowlist")
        if feature_file.num_row_groups != label_file.num_row_groups:
            raise ValueError("feature and label row-group counts do not reconcile")
        if tuple(label_file.schema_arrow.names) != ModelDatasetBuilder.LABEL_SCHEMA:
            raise ValueError("action label schema is not the Plan 04 contract")

        assignments = pq.read_table(splits_path)
        if tuple(assignments.column_names) != ModelDatasetBuilder.SPLIT_SCHEMA:
            raise ValueError("split assignment schema is not the Plan 04 contract")
        assignment_rows = assignments.to_pylist()
        split_by_match = {
            int(row["match_id"]): str(row["split"])
            for row in assignment_rows
        }
        if len(split_by_match) != len(assignment_rows):
            raise ValueError("split assignments contain duplicate match_id values")
        if set(split_by_match.values()) != set(SPLIT_NAMES):
            raise ValueError("split assignments contain invalid split names")
        if any(row["split_version"] != SPLIT_VERSION for row in assignment_rows):
            raise ValueError("split assignments use an unexpected version")

        paths = ChunkedModelDatasetPaths(
            directory=directory,
            model_dataset=directory / MODEL_DATASET_FILENAME,
            feature_allowlist=directory / FEATURE_ALLOWLIST_FILENAME,
            manifest=directory / MODEL_DATASET_MANIFEST_FILENAME,
        )
        temporary = paths.model_dataset.with_suffix(".parquet.tmp")
        temporary.unlink(missing_ok=True)
        writer = None
        eligible_count = 0
        seen_match_ids: set[int] = set()
        metadata = {
            b"model_dataset_version": MODEL_DATASET_VERSION.encode(),
            b"source_dataset_fingerprint": fingerprint.encode(),
            b"feature_version": BASE_FEATURE_VERSION.encode(),
            b"target_policy_version": TARGET_POLICY_VERSION.encode(),
            b"split_version": SPLIT_VERSION.encode(),
            b"feature_allowlist_sha256": self._canonical_sha256(
                list(allowlist)
            ).encode(),
        }

        try:
            for row_group in range(feature_file.num_row_groups):
                features = feature_file.read_row_group(row_group)
                labels = label_file.read_row_group(row_group)
                if features.num_rows != labels.num_rows:
                    raise ValueError(
                        f"feature/label row count mismatch in batch {row_group + 1}"
                    )
                for key in ("match_id", "action_id"):
                    if not features[key].equals(labels[key]):
                        raise ValueError(
                            f"feature/label {key} mismatch in batch {row_group + 1}"
                        )
                eligible = labels["eligible"]
                selected_features = features.filter(eligible)
                selected_labels = labels.filter(eligible)
                if any(
                    value != TARGET_POLICY_VERSION
                    for value in selected_labels["target_policy_version"].to_pylist()
                ):
                    raise ValueError("labels use an unexpected target policy version")
                if selected_labels["scores"].null_count or selected_labels[
                    "concedes"
                ].null_count:
                    raise ValueError("eligible labels cannot contain null targets")
                match_values = selected_features["match_id"].to_pylist()
                try:
                    split_values = [split_by_match[int(value)] for value in match_values]
                except KeyError as exc:
                    raise ValueError(
                        f"feature match has no split assignment: {exc.args[0]}"
                    ) from exc
                seen_match_ids.update(int(value) for value in match_values)
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
                table = pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(
                    metadata
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                eligible_count += table.num_rows
                if progress is not None:
                    progress(
                        f"joined batch {row_group + 1}/{feature_file.num_row_groups}: "
                        f"{table.num_rows} eligible rows"
                    )
            if writer is None:
                raise ValueError("feature artifact contains no row groups")
            writer.close()
            writer = None
            expected_eligible = sum(
                int(value["eligible_label_count"])
                for value in split_manifest["splits"].values()
            )
            if eligible_count != expected_eligible:
                raise ValueError("model dataset eligible-row count does not reconcile")
            if seen_match_ids != set(split_by_match):
                raise ValueError("model dataset match IDs do not reconcile with splits")
            temporary.replace(paths.model_dataset)

            allowlist_document = feature_allowlist_manifest(
                allowlist,
                source_dataset_fingerprint=fingerprint,
                feature_version=BASE_FEATURE_VERSION,
            )
            self._write_json(paths.feature_allowlist, allowlist_document)
            manifest = {
                "schema_version": 1,
                "model_dataset_version": MODEL_DATASET_VERSION,
                "source_dataset_fingerprint": fingerprint,
                "target_policy_version": TARGET_POLICY_VERSION,
                "split_version": SPLIT_VERSION,
                "row_count": eligible_count,
                "match_count": len(seen_match_ids),
                "feature_count": len(allowlist),
                "files": {
                    paths.model_dataset.name: self._sha256(paths.model_dataset),
                    paths.feature_allowlist.name: self._sha256(
                        paths.feature_allowlist
                    ),
                },
            }
            self._write_json(paths.manifest, manifest)
            if progress is not None:
                progress(f"completed model dataset: {paths.model_dataset}")
            return paths
        except BaseException:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def _verify_declared_hash(
        cls, path: Path, manifest: dict[str, Any], progress: Any, label: str
    ) -> None:
        expected = (manifest.get("files") or {}).get(path.name)
        if progress is not None:
            progress(f"verifying {label} hash")
        if not isinstance(expected, str) or cls._sha256(path) != expected:
            raise ValueError(f"{label} hash does not match its manifest")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}")
        return value

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

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True) + "\n"
        return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "MODEL_DATASET_MANIFEST_FILENAME",
    "ChunkedModelDatasetPaths",
    "ChunkedModelDatasetWriter",
]
