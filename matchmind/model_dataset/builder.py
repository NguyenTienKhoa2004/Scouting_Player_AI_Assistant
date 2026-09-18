"""Build the versioned, leakage-safe model-ready VAEP dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matchmind.vaep_features.feature_dataset_loader import (
    FeatureDataset,
)
from matchmind.labeling_and_splitting.splits import SPLIT_NAMES, SPLIT_VERSION
from matchmind.labeling_and_splitting.targets import LABEL_COLUMNS, TARGET_POLICY_VERSION
from .feature_allowlist import (
    FEATURE_ALLOWLIST_FILENAME,
    KNOWN_LEAKAGE_COLUMNS,
    MODEL_DATASET_FILENAME,
    MODEL_DATASET_SCHEMA_VERSION,
    MODEL_DATASET_VERSION,
    NON_FEATURE_COLUMNS,
    TRACE_COLUMNS,
    VERSION_COLUMNS,
    ModelDatasetError,
    feature_allowlist_manifest,
    select_model_matrix,
    validate_feature_allowlist,
)


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
        source_artifacts: FeatureDataset,
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

        manifest = feature_allowlist_manifest(
            feature_allowlist,
            source_dataset_fingerprint=source_artifacts.lineage.dataset_fingerprint,
            feature_version=source_artifacts.lineage.feature_version,
        )
        metadata = {
            b"model_dataset_version": MODEL_DATASET_VERSION.encode(),
            b"source_dataset_fingerprint": (
                source_artifacts.lineage.dataset_fingerprint.encode()
            ),
            b"feature_version": source_artifacts.lineage.feature_version.encode(),
            b"target_policy_version": TARGET_POLICY_VERSION.encode(),
            b"split_version": SPLIT_VERSION.encode(),
            b"feature_allowlist_sha256": manifest[
                "feature_allowlist_sha256"
            ].encode(),
        }
        table = pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(
            metadata
        )
        self._verify_output(table, feature_allowlist, len(indices))
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
