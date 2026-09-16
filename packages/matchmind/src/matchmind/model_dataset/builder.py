"""Build the versioned, leakage-safe model-ready VAEP dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from matchmind.vaep_features.artifact_loader import (
    FeatureArtifacts,
    baseline_feature_allowlist,
)
from matchmind.labeling_and_splitting.splits import SPLIT_NAMES, SPLIT_VERSION
from matchmind.labeling_and_splitting.targets import LABEL_COLUMNS, TARGET_POLICY_VERSION


MODEL_DATASET_VERSION = "vaep-model-dataset-v1"
MODEL_DATASET_SCHEMA_VERSION = 1
MODEL_DATASET_FILENAME = "model_dataset.parquet"
FEATURE_ALLOWLIST_FILENAME = "feature_allowlist.json"

TRACE_COLUMNS = (
    "analytics_run_id",
    "match_id",
    "action_id",
)
VERSION_COLUMNS = (
    "feature_version",
    "target_policy_version",
    "split_version",
)
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
    """Raised when source artifacts cannot form a safe model dataset."""


@dataclass(frozen=True, slots=True)
class ModelDataset:
    """A joined dataset plus the only columns permitted in model input."""

    table: Any
    feature_allowlist: tuple[str, ...]
    allowlist_manifest: dict[str, Any]

    def model_matrix(self, *, split: str | None = None) -> Any:
        """Return X using the explicit allowlist, never column inference."""

        return select_model_matrix(
            self.table,
            feature_allowlist=self.feature_allowlist,
            split=split,
        )


class ModelDatasetBuilder:
    """Join verified features, eligible labels, and complete-match splits."""

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

    def build(
        self,
        source_artifacts: FeatureArtifacts,
        labels: Any,
        split_assignments: Any,
    ) -> ModelDataset:
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError(
                "model dataset construction requires pyarrow; install requirements.txt"
            ) from exc

        feature_allowlist = validate_feature_allowlist(source_artifacts.feature_allowlist)
        self._require_schema(labels, self.LABEL_SCHEMA, "action labels")
        self._require_schema(
            split_assignments, self.SPLIT_SCHEMA, "split assignments"
        )

        feature_keys = self._keys(source_artifacts.features, "features")
        label_rows = self._label_rows(labels, source_artifacts.lineage.analytics_run_id)
        if set(feature_keys) != set(label_rows):
            raise ModelDatasetError(
                "feature and label (match_id, action_id) keys do not reconcile"
            )

        split_by_match = self._split_rows(split_assignments)
        feature_match_ids = {match_id for match_id, _ in feature_keys}
        if feature_match_ids != set(split_by_match):
            raise ModelDatasetError(
                "feature matches and split assignment match_id values do not reconcile"
            )

        indices: list[int] = []
        eligible_labels: list[dict[str, Any]] = []
        split_values: list[str] = []
        for index, key in enumerate(feature_keys):
            label = label_rows[key]
            if not label["eligible"]:
                continue
            indices.append(index)
            eligible_labels.append(label)
            split_values.append(split_by_match[key[0]]["split"])

        if not indices:
            raise ModelDatasetError("model dataset has no eligible rows")

        selected = source_artifacts.features.take(pa.array(indices, type=pa.int64()))
        arrays = [
            pa.array(
                [row["analytics_run_id"] for row in eligible_labels],
                type=pa.int64(),
            ),
            selected["match_id"],
            selected["action_id"],
        ]
        names = list(TRACE_COLUMNS)
        arrays.extend(selected[name] for name in feature_allowlist)
        names.extend(feature_allowlist)
        arrays.extend(
            [
                pa.array([row["scores"] for row in eligible_labels], type=pa.bool_()),
                pa.array(
                    [row["concedes"] for row in eligible_labels], type=pa.bool_()
                ),
                pa.array(split_values, type=pa.string()),
                selected["feature_version"],
                pa.array(
                    [row["target_policy_version"] for row in eligible_labels],
                    type=pa.string(),
                ),
                pa.array([SPLIT_VERSION] * len(indices), type=pa.string()),
            ]
        )
        names.extend(LABEL_COLUMNS + ("split",) + VERSION_COLUMNS)

        allowlist_sha256 = _canonical_sha256(list(feature_allowlist))
        metadata = {
            b"model_dataset_version": MODEL_DATASET_VERSION.encode(),
            b"source_dataset_fingerprint": (
                source_artifacts.lineage.dataset_fingerprint.encode()
            ),
            b"feature_version": source_artifacts.lineage.feature_version.encode(),
            b"target_policy_version": TARGET_POLICY_VERSION.encode(),
            b"split_version": SPLIT_VERSION.encode(),
            b"feature_allowlist_sha256": allowlist_sha256.encode(),
        }
        table = pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(
            metadata
        )
        self._verify_output(table, feature_allowlist, len(indices))
        manifest = feature_allowlist_manifest(
            feature_allowlist,
            source_dataset_fingerprint=source_artifacts.lineage.dataset_fingerprint,
            feature_version=source_artifacts.lineage.feature_version,
        )
        return ModelDataset(
            table=table,
            feature_allowlist=feature_allowlist,
            allowlist_manifest=manifest,
        )

    @staticmethod
    def _require_schema(table: Any, expected: tuple[str, ...], label: str) -> None:
        actual = tuple(table.column_names)
        if actual != expected:
            raise ModelDatasetError(
                f"{label} schema must be exactly {list(expected)}, got {list(actual)}"
            )

    @staticmethod
    def _keys(table: Any, label: str) -> list[tuple[int, int]]:
        keys = list(
            zip(
                table["match_id"].to_pylist(),
                table["action_id"].to_pylist(),
                strict=True,
            )
        )
        if any(
            isinstance(match_id, bool)
            or not isinstance(match_id, int)
            or isinstance(action_id, bool)
            or not isinstance(action_id, int)
            for match_id, action_id in keys
        ):
            raise ModelDatasetError(f"{label} keys must be non-null integers")
        if len(keys) != len(set(keys)):
            raise ModelDatasetError(f"{label} contain duplicate action keys")
        return keys

    @classmethod
    def _label_rows(
        cls, labels: Any, expected_analytics_run_id: int | None
    ) -> dict[tuple[int, int], dict[str, Any]]:
        keys = cls._keys(labels, "labels")
        rows = labels.to_pylist()
        result: dict[tuple[int, int], dict[str, Any]] = {}
        for key, row in zip(keys, rows, strict=True):
            if row["analytics_run_id"] != expected_analytics_run_id:
                raise ModelDatasetError(
                    "label analytics_run_id does not match the feature artifact lineage"
                )
            if row["target_policy_version"] != TARGET_POLICY_VERSION:
                raise ModelDatasetError("labels use an unexpected target policy version")
            if row["eligible"]:
                if row["scores"] is None or row["concedes"] is None:
                    raise ModelDatasetError("eligible labels cannot have null targets")
                if row["exclusion_reason"] is not None:
                    raise ModelDatasetError(
                        "eligible labels cannot have an exclusion reason"
                    )
            elif (
                row["scores"] is not None
                or row["concedes"] is not None
                or row["exclusion_reason"] is None
            ):
                raise ModelDatasetError(
                    "excluded labels require null targets and an exclusion reason"
                )
            result[key] = row
        return result

    @staticmethod
    def _split_rows(split_assignments: Any) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for row in split_assignments.to_pylist():
            match_id = row["match_id"]
            if isinstance(match_id, bool) or not isinstance(match_id, int):
                raise ModelDatasetError("split match_id values must be non-null integers")
            if match_id in result:
                raise ModelDatasetError("split assignments contain duplicate match_id")
            if row["split"] not in SPLIT_NAMES:
                raise ModelDatasetError(f"invalid split name: {row['split']!r}")
            if row["split_version"] != SPLIT_VERSION:
                raise ModelDatasetError("split assignments use an unexpected version")
            result[match_id] = row
        return result

    @staticmethod
    def _verify_output(
        table: Any, feature_allowlist: tuple[str, ...], eligible_count: int
    ) -> None:
        expected = TRACE_COLUMNS + feature_allowlist + LABEL_COLUMNS + (
            "split",
        ) + VERSION_COLUMNS
        if tuple(table.column_names) != expected:
            raise ModelDatasetError("model dataset column order is not deterministic")
        if table.num_rows != eligible_count:
            raise ModelDatasetError("model dataset row count does not reconcile")
        matrix = select_model_matrix(table, feature_allowlist=feature_allowlist)
        if tuple(matrix.column_names) != feature_allowlist:
            raise ModelDatasetError("model matrix contains non-feature columns")


def validate_feature_allowlist(columns: Iterable[str]) -> tuple[str, ...]:
    """Require the exact baseline contract and reject identifiers/metadata."""

    actual = tuple(columns)
    expected = baseline_feature_allowlist()
    if actual != expected:
        raise ModelDatasetError(
            "feature allowlist must exactly match the declared socceraction baseline"
        )
    forbidden = NON_FEATURE_COLUMNS | KNOWN_LEAKAGE_COLUMNS
    leaked = sorted(set(actual) & forbidden)
    if leaked:
        raise ModelDatasetError(
            f"feature allowlist contains identifiers or metadata: {leaked}"
        )
    if any(name.endswith("_a3") for name in actual):
        raise ModelDatasetError("feature allowlist cannot contain a3 columns")
    return actual


def select_model_matrix(
    model_dataset: Any,
    *,
    feature_allowlist: Iterable[str],
    split: str | None = None,
) -> Any:
    """Select model inputs only from the validated, explicit feature contract."""

    allowlist = validate_feature_allowlist(feature_allowlist)
    missing = set(allowlist) - set(model_dataset.column_names)
    if missing:
        raise ModelDatasetError(
            f"model dataset is missing allowlisted features: {sorted(missing)}"
        )
    selected = model_dataset
    if split is not None:
        if split not in SPLIT_NAMES:
            raise ModelDatasetError(f"invalid split name: {split!r}")
        if "split" not in selected.column_names:
            raise ModelDatasetError("model dataset has no split column")
        try:
            import pyarrow.compute as pc
        except ImportError as exc:
            raise RuntimeError(
                "model matrix selection requires pyarrow; install requirements.txt"
            ) from exc
        selected = selected.filter(pc.equal(selected["split"], split))
    return selected.select(list(allowlist))


def feature_allowlist_manifest(
    feature_allowlist: Iterable[str],
    *,
    source_dataset_fingerprint: str,
    feature_version: str,
) -> dict[str, Any]:
    """Describe and attest the only columns permitted to enter fitted models."""

    allowlist = validate_feature_allowlist(feature_allowlist)
    return {
        "schema_version": MODEL_DATASET_SCHEMA_VERSION,
        "model_dataset_version": MODEL_DATASET_VERSION,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "feature_version": feature_version,
        "feature_count": len(allowlist),
        "feature_columns": list(allowlist),
        "feature_allowlist_sha256": _canonical_sha256(list(allowlist)),
        "selection_policy": "explicit_allowlist_only",
        "excluded_column_groups": {
            "identifiers": list(TRACE_COLUMNS),
            "targets": list(LABEL_COLUMNS),
            "split": ["split"],
            "versions": list(VERSION_COLUMNS),
        },
        "checks": {
            "exact_socceraction_baseline_allowlist": True,
            "identifiers_or_metadata_in_model_matrix": False,
            "a3_columns_present": False,
        },
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "FEATURE_ALLOWLIST_FILENAME",
    "KNOWN_LEAKAGE_COLUMNS",
    "MODEL_DATASET_FILENAME",
    "MODEL_DATASET_SCHEMA_VERSION",
    "MODEL_DATASET_VERSION",
    "NON_FEATURE_COLUMNS",
    "TRACE_COLUMNS",
    "VERSION_COLUMNS",
    "ModelDataset",
    "ModelDatasetBuilder",
    "ModelDatasetError",
    "feature_allowlist_manifest",
    "select_model_matrix",
    "validate_feature_allowlist",
]
