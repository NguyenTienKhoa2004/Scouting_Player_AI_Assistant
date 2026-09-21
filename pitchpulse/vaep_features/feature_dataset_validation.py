"""Validation rules for a saved VAEP feature dataset."""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from socceraction.vaep import VAEP
from socceraction.vaep.features import feature_column_names

from pitchpulse.spadl.converter import (
    ACTION_MAPPING_VERSION,
    COORDINATE_SYSTEM_VERSION,
)

from .action_state import STATE_CONTRACT_VERSION
from .feature_builder import BASE_FEATURE_VERSION, NB_PREV_ACTIONS


ACTION_FILENAME = "actions.parquet"
FEATURE_FILENAME = "action_features.parquet"
QUALITY_REPORT_FILENAME = "conversion_quality_report.json"
MANIFEST_FILENAME = "manifest.json"
FEATURE_METADATA_COLUMNS = ("match_id", "action_id", "feature_version")
REQUIRED_ACTION_COLUMNS = frozenset(
    {
        "match_id",
        "action_id",
        "source",
        "source_event_id",
        "source_event_index",
        "source_action_index",
        "mapping_version",
        "coordinate_system_version",
        "state_version",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        "type_id",
        "type_name",
        "result_id",
        "result_name",
        "is_shootout",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FeatureDatasetValidationError(ValueError):
    """Raised when a feature dataset cannot safely enter model training."""


@dataclass(frozen=True, slots=True)
class FeatureDatasetLineage:
    analytics_run_id: int | None
    dataset_fingerprint: str
    mapping_version: str
    coordinate_system_version: str
    state_contract_version: str
    feature_version: str


@lru_cache(maxsize=1)
def baseline_feature_allowlist() -> tuple[str, ...]:
    """Return socceraction's default a0/a1/a2 feature names."""

    if NB_PREV_ACTIONS != 3:
        raise FeatureDatasetValidationError(
            f"baseline requires three actions, configured {NB_PREV_ACTIONS}"
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=r"socceraction\..*"
        )
        names = tuple(
            str(name)
            for name in feature_column_names(
                VAEP(nb_prev_actions=NB_PREV_ACTIONS).xfns,
                nb_prev_actions=NB_PREV_ACTIONS,
            )
        )
    if len(names) != len(set(names)):
        raise FeatureDatasetValidationError("baseline feature list is not unique")
    if any(name.endswith("_a3") for name in names):
        raise FeatureDatasetValidationError("baseline feature list contains a3")
    return names


def validate_lineage(
    directory: Path,
    manifest: dict[str, Any],
    quality_report: dict[str, Any],
    *,
    expected_dataset_fingerprint: str | None,
    expected_analytics_run_id: int | None,
) -> FeatureDatasetLineage:
    expected_versions = {
        "mapping_version": ACTION_MAPPING_VERSION,
        "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
        "state_contract_version": STATE_CONTRACT_VERSION,
        "feature_version": BASE_FEATURE_VERSION,
    }
    for field, expected in expected_versions.items():
        quality_value = quality_report.get(field)
        if quality_value != expected:
            raise FeatureDatasetValidationError(
                f"quality report {field} must be {expected!r}, got "
                f"{quality_value!r}"
            )
        if field == "state_contract_version" and field not in manifest:
            continue
        manifest_value = manifest.get(field)
        if manifest_value != expected:
            raise FeatureDatasetValidationError(
                f"manifest {field} must be {expected!r}, got {manifest_value!r}"
            )

    if manifest.get("include_360") is not False:
        raise FeatureDatasetValidationError(
            "baseline modeling requires include_360=false"
        )
    if quality_report.get("include_360") is not False:
        raise FeatureDatasetValidationError(
            "quality report does not describe a baseline feature dataset"
        )

    fingerprint = manifest.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256_PATTERN.fullmatch(fingerprint):
        raise FeatureDatasetValidationError(
            "manifest dataset_fingerprint must be a lowercase SHA-256"
        )
    if (
        expected_dataset_fingerprint is not None
        and fingerprint != expected_dataset_fingerprint
    ):
        raise FeatureDatasetValidationError(
            "dataset fingerprint does not match the requested lineage"
        )
    if directory.name.startswith("dataset-") and directory.name != (
        f"dataset-{fingerprint[:16]}"
    ):
        raise FeatureDatasetValidationError(
            "artifact directory does not match dataset fingerprint"
        )

    analytics_run_id = manifest.get("analytics_run_id")
    if analytics_run_id is not None and (
        isinstance(analytics_run_id, bool)
        or not isinstance(analytics_run_id, int)
        or analytics_run_id <= 0
    ):
        raise FeatureDatasetValidationError(
            "manifest analytics_run_id must be a positive integer"
        )
    if (
        expected_analytics_run_id is not None
        and analytics_run_id != expected_analytics_run_id
    ):
        raise FeatureDatasetValidationError(
            "analytics run does not match the requested lineage"
        )

    return FeatureDatasetLineage(
        analytics_run_id=analytics_run_id,
        dataset_fingerprint=fingerprint,
        mapping_version=ACTION_MAPPING_VERSION,
        coordinate_system_version=COORDINATE_SYSTEM_VERSION,
        state_contract_version=STATE_CONTRACT_VERSION,
        feature_version=BASE_FEATURE_VERSION,
    )


def validate_schemas(
    action_schema: Any,
    feature_schema: Any,
    lineage: FeatureDatasetLineage,
) -> None:
    missing_action_columns = REQUIRED_ACTION_COLUMNS - set(action_schema.names)
    if missing_action_columns:
        raise FeatureDatasetValidationError(
            f"actions schema is missing columns: {sorted(missing_action_columns)}"
        )

    expected_feature_columns = FEATURE_METADATA_COLUMNS + baseline_feature_allowlist()
    if tuple(feature_schema.names) != expected_feature_columns:
        actual = set(feature_schema.names)
        expected = set(expected_feature_columns)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        order_mismatch = not missing and not unexpected
        raise FeatureDatasetValidationError(
            "feature schema does not match the exact baseline allowlist; "
            f"missing={missing}, unexpected={unexpected}, "
            f"order_mismatch={order_mismatch}"
        )

    expected_metadata = {
        "mapping_version": lineage.mapping_version,
        "coordinate_system_version": lineage.coordinate_system_version,
        "feature_version": lineage.feature_version,
    }
    for label, schema in (
        (ACTION_FILENAME, action_schema),
        (FEATURE_FILENAME, feature_schema),
    ):
        metadata = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in (schema.metadata or {}).items()
        }
        for field, expected in expected_metadata.items():
            if metadata.get(field) != expected:
                raise FeatureDatasetValidationError(
                    f"{label} metadata {field} must be {expected!r}"
                )
        optional_metadata = {
            "state_contract_version": lineage.state_contract_version,
            "dataset_fingerprint": lineage.dataset_fingerprint,
        }
        for field, expected in optional_metadata.items():
            if field in metadata and metadata[field] != expected:
                raise FeatureDatasetValidationError(
                    f"{label} metadata {field} must be {expected!r}"
                )


def validate_rows(
    actions: Any,
    features: Any,
    quality_report: dict[str, Any],
    lineage: FeatureDatasetLineage,
) -> None:
    action_keys = _keys(actions, ACTION_FILENAME)
    feature_keys = _keys(features, FEATURE_FILENAME)
    if len(action_keys) != len(feature_keys):
        raise FeatureDatasetValidationError(
            "action and feature row counts do not reconcile"
        )
    if len(set(action_keys)) != len(action_keys):
        raise FeatureDatasetValidationError("actions contain duplicate keys")
    if len(set(feature_keys)) != len(feature_keys):
        raise FeatureDatasetValidationError("features contain duplicate keys")
    if set(action_keys) != set(feature_keys):
        raise FeatureDatasetValidationError(
            "action and feature (match_id, action_id) keys do not reconcile"
        )

    ids_by_match: defaultdict[int, list[int]] = defaultdict(list)
    for match_id, action_id in action_keys:
        ids_by_match[match_id].append(action_id)
    for match_id, action_ids in ids_by_match.items():
        if action_ids != list(range(len(action_ids))):
            raise FeatureDatasetValidationError(
                f"match {match_id} action_id must be contiguous and zero-based"
            )

    _require_single_value(
        actions, "mapping_version", lineage.mapping_version, ACTION_FILENAME
    )
    _require_single_value(
        actions,
        "coordinate_system_version",
        lineage.coordinate_system_version,
        ACTION_FILENAME,
    )
    _require_single_value(
        actions, "state_version", lineage.state_contract_version, ACTION_FILENAME
    )
    _require_single_value(
        features, "feature_version", lineage.feature_version, FEATURE_FILENAME
    )

    expected_counts = {
        "action_count": len(action_keys),
        "state_count": len(action_keys),
        "feature_count": len(feature_keys),
        "match_count": len(ids_by_match),
    }
    for field, expected in expected_counts.items():
        if quality_report.get(field) != expected:
            raise FeatureDatasetValidationError(
                f"quality report {field} must be {expected}, got "
                f"{quality_report.get(field)!r}"
            )


def _keys(table: Any, label: str) -> list[tuple[int, int]]:
    match_ids = table["match_id"].to_pylist()
    action_ids = table["action_id"].to_pylist()
    keys: list[tuple[int, int]] = []
    for match_id, action_id in zip(match_ids, action_ids, strict=True):
        if (
            isinstance(match_id, bool)
            or not isinstance(match_id, int)
            or isinstance(action_id, bool)
            or not isinstance(action_id, int)
        ):
            raise FeatureDatasetValidationError(
                f"{label} keys must be non-null integers"
            )
        keys.append((match_id, action_id))
    return keys


def _require_single_value(
    table: Any,
    column: str,
    expected: str,
    label: str,
) -> None:
    actual = set(table[column].to_pylist())
    if actual != {expected}:
        raise FeatureDatasetValidationError(
            f"{label} column {column} must contain only {expected!r}, got "
            f"{sorted(str(value) for value in actual)}"
        )
