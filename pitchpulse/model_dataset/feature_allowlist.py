"""Define the columns that may enter a VAEP model."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from pitchpulse.labeling_and_splitting.targets import LABEL_COLUMNS
from pitchpulse.vaep_features.feature_dataset_loader import baseline_feature_allowlist


MODEL_DATASET_VERSION = "vaep-model-dataset-v1"
MODEL_DATASET_SCHEMA_VERSION = 1
MODEL_DATASET_FILENAME = "model_dataset.parquet"
FEATURE_ALLOWLIST_FILENAME = "feature_allowlist.json"

TRACE_COLUMNS = ("analytics_run_id", "match_id", "action_id")
VERSION_COLUMNS = ("feature_version", "target_policy_version", "split_version")
NON_FEATURE_COLUMNS = frozenset(
    TRACE_COLUMNS + LABEL_COLUMNS + ("split",) + VERSION_COLUMNS
)
KNOWN_LEAKAGE_COLUMNS = frozenset(
    {
        "competition_id",
        "season_id",
        "match_date",
        "split_order",
        "source_event_id",
        "source_action_index",
        "source_event_index",
        "player_id",
        "player_name",
        "team_id",
        "run_id",
        "analytics_run_id",
        "final_home_score",
        "final_away_score",
        "future_possession_outcome",
        "future_360_frame",
    }
)


class ModelDatasetError(ValueError):
    """Raised when source data cannot form a safe model dataset."""


def validate_feature_allowlist(columns: Iterable[str]) -> tuple[str, ...]:
    actual = tuple(columns)
    expected = baseline_feature_allowlist()
    if actual != expected:
        raise ModelDatasetError(
            "Feature list must exactly match the standard VAEP feature list"
        )
    leaked = sorted(set(actual) & (NON_FEATURE_COLUMNS | KNOWN_LEAKAGE_COLUMNS))
    if leaked:
        raise ModelDatasetError(f"Feature list contains identifiers: {leaked}")
    if any(name.endswith("_a3") for name in actual):
        raise ModelDatasetError("Feature list cannot contain a3 columns")
    return actual


def feature_allowlist_manifest(
    feature_allowlist: Iterable[str],
    *,
    source_dataset_fingerprint: str,
    feature_version: str,
) -> dict[str, Any]:
    allowlist = validate_feature_allowlist(feature_allowlist)
    payload = json.dumps(list(allowlist), separators=(",", ":")) + "\n"
    return {
        "schema_version": MODEL_DATASET_SCHEMA_VERSION,
        "model_dataset_version": MODEL_DATASET_VERSION,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "feature_version": feature_version,
        "feature_count": len(allowlist),
        "feature_columns": list(allowlist),
        "feature_allowlist_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


__all__ = [
    "FEATURE_ALLOWLIST_FILENAME",
    "KNOWN_LEAKAGE_COLUMNS",
    "MODEL_DATASET_FILENAME",
    "MODEL_DATASET_SCHEMA_VERSION",
    "MODEL_DATASET_VERSION",
    "NON_FEATURE_COLUMNS",
    "TRACE_COLUMNS",
    "VERSION_COLUMNS",
    "ModelDatasetError",
    "feature_allowlist_manifest",
    "validate_feature_allowlist",
]
