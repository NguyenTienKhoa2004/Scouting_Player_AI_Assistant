"""Save generated actions and VAEP features as Parquet files."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matchmind.shared.file_io import file_sha256, write_json
from matchmind.spadl.converter import SpadlConversionReport

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
    """Write one feature dataset and its small JSON reports."""

    def write(self, dataset: AnalyticsDataset, output_root: Path) -> ArtifactPaths:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requires pyarrow; install requirements.txt"
            ) from exc

        fingerprint = self._dataset_fingerprint(dataset)
        directory = (
            Path(output_root)
            / dataset.conversion_report.mapping_version
            / dataset.feature_version
            / f"dataset-{fingerprint[:16]}"
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
            b"dataset_fingerprint": fingerprint.encode(),
        }
        actions = pa.Table.from_pylist(
            [state.as_dict() for state in dataset.states]
        ).replace_schema_metadata(metadata)
        features = pa.Table.from_pylist(
            [row.as_dict() for row in dataset.features]
        ).replace_schema_metadata(metadata)
        pq.write_table(actions, paths.actions, compression="zstd")
        pq.write_table(features, paths.features, compression="zstd")
        write_json(paths.quality_report, dataset.quality_report())
        write_json(
            paths.manifest,
            {
                "mapping_version": dataset.conversion_report.mapping_version,
                "coordinate_system_version": (
                    dataset.conversion_report.coordinate_system_version
                ),
                "state_contract_version": STATE_CONTRACT_VERSION,
                "feature_version": dataset.feature_version,
                "include_360": dataset.include_360,
                "dataset_fingerprint": fingerprint,
                "files": {
                    paths.actions.name: file_sha256(paths.actions),
                    paths.features.name: file_sha256(paths.features),
                    paths.quality_report.name: file_sha256(paths.quality_report),
                },
            },
        )
        return paths

    def write_many(
        self,
        datasets: Iterable[AnalyticsDataset],
        output_root: Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> ArtifactPaths:
        """Combine several batches, then write them like one dataset."""

        combined = _combine_datasets(list(datasets))
        if progress is not None:
            progress(f"writing {combined.conversion_report.match_count} matches")
        paths = self.write(combined, output_root)
        if progress is not None:
            progress(f"completed feature dataset: {paths.directory}")
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


def _combine_datasets(datasets: list[AnalyticsDataset]) -> AnalyticsDataset:
    if not datasets:
        raise ValueError("cannot write an empty feature dataset")

    first = datasets[0]
    expected = (
        first.feature_version,
        first.include_360,
        first.conversion_report.mapping_version,
        first.conversion_report.coordinate_system_version,
    )
    seen_matches: set[int] = set()
    exclusions: Counter[str] = Counter()
    warnings: Counter[str] = Counter()

    for dataset in datasets:
        actual = (
            dataset.feature_version,
            dataset.include_360,
            dataset.conversion_report.mapping_version,
            dataset.conversion_report.coordinate_system_version,
        )
        if actual != expected:
            raise ValueError("all feature batches must use the same versions")
        match_ids = {action.match_id for action in dataset.actions}
        overlap = seen_matches & match_ids
        if overlap:
            raise ValueError(
                f"feature batches contain duplicate matches: {sorted(overlap)}"
            )
        seen_matches.update(match_ids)
        exclusions.update(dict(dataset.conversion_report.exclusions_by_reason))
        warnings.update(dict(dataset.input_warnings_by_code))

    reports = [dataset.conversion_report for dataset in datasets]
    report = SpadlConversionReport(
        mapping_version=expected[2],
        coordinate_system_version=expected[3],
        match_count=len(seen_matches),
        event_count=sum(item.event_count for item in reports),
        mapped_event_count=sum(item.mapped_event_count for item in reports),
        excluded_event_count=sum(item.excluded_event_count for item in reports),
        action_count=sum(item.action_count for item in reports),
        synthetic_action_count=sum(item.synthetic_action_count for item in reports),
        exclusions_by_reason=tuple(sorted(exclusions.items())),
    )
    return AnalyticsDataset(
        actions=tuple(action for item in datasets for action in item.actions),
        states=tuple(state for item in datasets for state in item.states),
        features=tuple(feature for item in datasets for feature in item.features),
        conversion_report=report,
        feature_version=first.feature_version,
        include_360=first.include_360,
        input_warning_count=sum(item.input_warning_count for item in datasets),
        input_warnings_by_code=tuple(sorted(warnings.items())),
    )


__all__ = ["ArtifactPaths", "ParquetArtifactWriter"]
