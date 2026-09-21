"""Load a saved VAEP feature dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pitchpulse.shared.file_io import read_json

from .feature_dataset_validation import (
    ACTION_FILENAME,
    FEATURE_FILENAME,
    FEATURE_METADATA_COLUMNS,
    MANIFEST_FILENAME,
    QUALITY_REPORT_FILENAME,
    REQUIRED_ACTION_COLUMNS,
    FeatureDatasetLineage,
    FeatureDatasetValidationError,
    baseline_feature_allowlist,
    validate_lineage,
    validate_rows,
    validate_schemas,
)


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    directory: Path
    actions: Any
    features: Any
    manifest: dict[str, Any]
    quality_report: dict[str, Any]
    lineage: FeatureDatasetLineage
    feature_allowlist: tuple[str, ...]


class ValidatedFeatureDatasetLoader:
    """Read a feature dataset and apply its validation rules."""

    def load(
        self,
        feature_dataset_directory: Path,
        *,
        expected_dataset_fingerprint: str | None = None,
        expected_analytics_run_id: int | None = None,
    ) -> FeatureDataset:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "feature dataset loading requires pyarrow; install requirements.txt"
            ) from exc

        directory = Path(feature_dataset_directory).resolve()
        if not directory.is_dir():
            raise FeatureDatasetValidationError(
                f"feature dataset directory does not exist: {directory}"
            )

        action_path = directory / ACTION_FILENAME
        feature_path = directory / FEATURE_FILENAME
        quality_path = directory / QUALITY_REPORT_FILENAME
        for path in (action_path, feature_path, quality_path):
            if not path.is_file():
                raise FeatureDatasetValidationError(
                    f"Feature dataset file is missing: {path}"
                )

        manifest = read_json(directory / MANIFEST_FILENAME)
        quality_report = read_json(quality_path)
        lineage = validate_lineage(
            directory,
            manifest,
            quality_report,
            expected_dataset_fingerprint=expected_dataset_fingerprint,
            expected_analytics_run_id=expected_analytics_run_id,
        )

        action_schema = pq.read_schema(action_path)
        feature_schema = pq.read_schema(feature_path)
        validate_schemas(action_schema, feature_schema, lineage)

        with action_path.open("rb") as source:
            actions = pq.read_table(source)
        with feature_path.open("rb") as source:
            features = pq.read_table(source)
        validate_rows(actions, features, quality_report, lineage)

        return FeatureDataset(
            directory=directory,
            actions=actions,
            features=features,
            manifest=manifest,
            quality_report=quality_report,
            lineage=lineage,
            feature_allowlist=baseline_feature_allowlist(),
        )


__all__ = [
    "ACTION_FILENAME",
    "FEATURE_FILENAME",
    "FEATURE_METADATA_COLUMNS",
    "MANIFEST_FILENAME",
    "FeatureDatasetValidationError",
    "ValidatedFeatureDatasetLoader",
    "FeatureDataset",
    "FeatureDatasetLineage",
    "QUALITY_REPORT_FILENAME",
    "REQUIRED_ACTION_COLUMNS",
    "baseline_feature_allowlist",
]
