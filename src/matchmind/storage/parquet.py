"""Deterministic Parquet and JSON artifacts for Plan 03 outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matchmind.feature_engineering.pipeline import AnalyticsDataset


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
            b"feature_version": dataset.feature_version.encode(),
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


__all__ = ["ArtifactPaths", "ParquetArtifactWriter"]
