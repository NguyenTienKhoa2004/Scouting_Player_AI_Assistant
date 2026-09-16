"""Deterministic Parquet and JSON artifacts for Plan 03 outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_state import STATE_CONTRACT_VERSION
from .builder import AnalyticsDataset


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    directory: Path
    actions: Path
    features: Path
    quality_report: Path
    manifest: Path


class ParquetArtifactWriter:
    def write(self, dataset: AnalyticsDataset, output_root: Path) -> ArtifactPaths:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requires pyarrow; install requirements.txt"
            ) from exc

        dataset_fingerprint = self._dataset_fingerprint(dataset)
        directory = (
            output_root
            / dataset.conversion_report.mapping_version
            / dataset.feature_version
            / f"dataset-{dataset_fingerprint[:16]}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        paths = ArtifactPaths(
            directory=directory,
            actions=directory / "actions.parquet",
            features=directory / "action_features.parquet",
            quality_report=directory / "conversion_quality_report.json",
            manifest=directory / "manifest.json",
        )
        metadata = {
            b"mapping_version": dataset.conversion_report.mapping_version.encode(),
            b"coordinate_system_version": (
                dataset.conversion_report.coordinate_system_version.encode()
            ),
            b"state_contract_version": STATE_CONTRACT_VERSION.encode(),
            b"feature_version": dataset.feature_version.encode(),
            b"dataset_fingerprint": dataset_fingerprint.encode(),
        }
        action_table = pa.Table.from_pylist(
            [state.as_dict() for state in dataset.states]
        ).replace_schema_metadata(metadata)
        feature_table = pa.Table.from_pylist(
            [row.as_dict() for row in dataset.features]
        ).replace_schema_metadata(metadata)
        self._write_parquet(pq, action_table, paths.actions)
        self._write_parquet(pq, feature_table, paths.features)
        self._write_json(paths.quality_report, dataset.quality_report())
        manifest = {
            "mapping_version": dataset.conversion_report.mapping_version,
            "coordinate_system_version": (
                dataset.conversion_report.coordinate_system_version
            ),
            "state_contract_version": STATE_CONTRACT_VERSION,
            "feature_version": dataset.feature_version,
            "include_360": dataset.include_360,
            "dataset_fingerprint": dataset_fingerprint,
            "files": {
                paths.actions.name: self._sha256(paths.actions),
                paths.features.name: self._sha256(paths.features),
                paths.quality_report.name: self._sha256(paths.quality_report),
            },
        }
        self._write_json(paths.manifest, manifest)
        return paths

    @staticmethod
    def _dataset_fingerprint(dataset: AnalyticsDataset) -> str:
        digest = hashlib.sha256()
        digest.update(dataset.conversion_report.mapping_version.encode())
        digest.update(dataset.feature_version.encode())
        digest.update(str(dataset.conversion_report.event_count).encode())
        for action in dataset.actions:
            digest.update(
                (
                    f"{action.match_id}|{action.source}|{action.source_event_id}|"
                    f"{action.source_event_index}|{action.source_action_index}\n"
                ).encode()
            )
        return digest.hexdigest()

    @staticmethod
    def _write_parquet(pq: Any, table: Any, target: Path) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(target)

    @staticmethod
    def _write_json(target: Path, value: dict[str, Any]) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class ChunkedParquetArtifactWriter:
    """Write one immutable Plan 03 artifact while keeping one batch in RAM."""

    _SUM_FIELDS = (
        "match_count",
        "event_count",
        "mapped_event_count",
        "excluded_event_count",
        "action_count",
        "synthetic_action_count",
        "input_warning_count",
        "state_count",
        "feature_count",
        "actions_with_360",
        "actions_without_360",
        "possession_transition_count",
        "missing_player_count",
        "coordinate_violation_count",
    )
    _COUNTER_FIELDS = ("exclusions_by_reason", "input_warnings_by_code")
    _CONSTANT_FIELDS = (
        "mapping_version",
        "coordinate_system_version",
        "state_contract_version",
        "feature_version",
        "include_360",
    )

    def write(
        self,
        datasets: Iterable[AnalyticsDataset],
        output_root: Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> ArtifactPaths:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requires pyarrow; install requirements.txt"
            ) from exc

        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=".plan03-staging-", dir=output_root)
        )
        batch_directory = staging / "batches"
        batch_directory.mkdir()
        action_batches: list[Path] = []
        feature_batches: list[Path] = []
        totals = {field: 0 for field in self._SUM_FIELDS}
        counters = {field: Counter() for field in self._COUNTER_FIELDS}
        constants: dict[str, Any] | None = None
        seen_match_ids: set[int] = set()

        try:
            for batch_index, dataset in enumerate(datasets, start=1):
                self._validate_dataset(dataset)
                report = dataset.quality_report()
                current_constants = {
                    field: report[field] for field in self._CONSTANT_FIELDS
                }
                if constants is None:
                    constants = current_constants
                elif current_constants != constants:
                    raise ValueError(
                        "all Plan 03 batches must use identical version contracts"
                    )

                match_ids = {action.match_id for action in dataset.actions}
                overlap = seen_match_ids & match_ids
                if overlap:
                    raise ValueError(
                        f"Plan 03 batches contain duplicate matches: {sorted(overlap)}"
                    )
                seen_match_ids.update(match_ids)

                for field in self._SUM_FIELDS:
                    totals[field] += int(report[field])
                for field in self._COUNTER_FIELDS:
                    counters[field].update(report[field])

                action_path = batch_directory / f"actions-{batch_index:05d}.parquet"
                feature_path = (
                    batch_directory / f"features-{batch_index:05d}.parquet"
                )
                pq.write_table(
                    pa.Table.from_pylist(
                        [state.as_dict() for state in dataset.states]
                    ),
                    action_path,
                    compression="zstd",
                )
                pq.write_table(
                    pa.Table.from_pylist(
                        [row.as_dict() for row in dataset.features]
                    ),
                    feature_path,
                    compression="zstd",
                )
                action_batches.append(action_path)
                feature_batches.append(feature_path)
                if progress is not None:
                    progress(
                        f"saved batch {batch_index}: {len(match_ids)} matches, "
                        f"{len(dataset.actions)} actions"
                    )

            if constants is None:
                raise ValueError("cannot write an empty Plan 03 dataset")
            if totals["match_count"] != len(seen_match_ids):
                raise ValueError(
                    "aggregated match count does not match unique batch matches"
                )

            metadata = {
                b"mapping_version": str(constants["mapping_version"]).encode(),
                b"coordinate_system_version": str(
                    constants["coordinate_system_version"]
                ).encode(),
                b"state_contract_version": str(
                    constants["state_contract_version"]
                ).encode(),
                b"feature_version": str(constants["feature_version"]).encode(),
            }
            staged_actions = staging / "actions.parquet"
            staged_features = staging / "action_features.parquet"
            if progress is not None:
                progress("consolidating action row groups")
            self._consolidate(pa, pq, action_batches, staged_actions, metadata)
            if progress is not None:
                progress("consolidating feature row groups")
            self._consolidate(pa, pq, feature_batches, staged_features, metadata)

            quality_report = {
                **constants,
                **totals,
                **{
                    field: dict(sorted(counter.items()))
                    for field, counter in counters.items()
                },
            }
            dataset_fingerprint = self._dataset_fingerprint(
                pq,
                staged_actions,
                mapping_version=str(constants["mapping_version"]),
                feature_version=str(constants["feature_version"]),
                event_count=totals["event_count"],
            )
            directory = (
                output_root
                / str(constants["mapping_version"])
                / str(constants["feature_version"])
                / f"dataset-{dataset_fingerprint[:16]}"
            )
            paths = ArtifactPaths(
                directory=directory,
                actions=directory / staged_actions.name,
                features=directory / staged_features.name,
                quality_report=directory / "conversion_quality_report.json",
                manifest=directory / "manifest.json",
            )
            quality_path = staging / paths.quality_report.name
            manifest_path = staging / paths.manifest.name
            ParquetArtifactWriter._write_json(quality_path, quality_report)
            manifest = {
                **constants,
                "dataset_fingerprint": dataset_fingerprint,
                "files": {
                    staged_actions.name: ParquetArtifactWriter._sha256(
                        staged_actions
                    ),
                    staged_features.name: ParquetArtifactWriter._sha256(
                        staged_features
                    ),
                    quality_path.name: ParquetArtifactWriter._sha256(quality_path),
                },
            }
            ParquetArtifactWriter._write_json(manifest_path, manifest)
            shutil.rmtree(batch_directory)

            directory.parent.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                existing_manifest = directory / "manifest.json"
                if existing_manifest.is_file():
                    existing = json.loads(
                        existing_manifest.read_text(encoding="utf-8")
                    )
                    if existing.get("dataset_fingerprint") == dataset_fingerprint:
                        shutil.rmtree(staging)
                        return paths
                raise FileExistsError(
                    f"artifact directory already exists but is incomplete: {directory}"
                )
            staging.replace(directory)
            if progress is not None:
                progress(f"completed Plan 03 artifact: {directory}")
            return paths
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _validate_dataset(dataset: AnalyticsDataset) -> None:
        if not dataset.actions:
            raise ValueError("a Plan 03 batch contains no actions")
        if len(dataset.actions) != len(dataset.states):
            raise ValueError("every action must have exactly one state")
        if len(dataset.actions) != len(dataset.features):
            raise ValueError("every action must have exactly one feature row")

    @staticmethod
    def _consolidate(
        pa: Any,
        pq: Any,
        batches: list[Path],
        target: Path,
        metadata: dict[bytes, bytes],
    ) -> None:
        schemas = [pq.read_schema(path).remove_metadata() for path in batches]
        schema = pa.unify_schemas(schemas, promote_options="permissive")
        output_schema = schema.with_metadata(metadata)
        with pq.ParquetWriter(target, output_schema, compression="zstd") as writer:
            for path in batches:
                table = pq.read_table(path).cast(schema).replace_schema_metadata(
                    metadata
                )
                writer.write_table(table)

    @staticmethod
    def _dataset_fingerprint(
        pq: Any,
        actions_path: Path,
        *,
        mapping_version: str,
        feature_version: str,
        event_count: int,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(mapping_version.encode())
        digest.update(feature_version.encode())
        digest.update(str(event_count).encode())
        columns = (
            "match_id",
            "source",
            "source_event_id",
            "source_event_index",
            "source_action_index",
        )
        parquet_file = pq.ParquetFile(actions_path)
        for batch in parquet_file.iter_batches(columns=list(columns)):
            values = batch.to_pydict()
            for row in zip(*(values[column] for column in columns), strict=True):
                digest.update(
                    ("|".join(str(value) for value in row) + "\n").encode()
                )
        return digest.hexdigest()


__all__ = [
    "ArtifactPaths",
    "ChunkedParquetArtifactWriter",
    "ParquetArtifactWriter",
]
