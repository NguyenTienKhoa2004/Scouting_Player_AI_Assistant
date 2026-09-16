"""Validated boundary between immutable feature engineering and model training."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from socceraction.vaep import VAEP
from socceraction.vaep.features import feature_column_names

from matchmind.analytics.features.action_state import STATE_CONTRACT_VERSION
from matchmind.analytics.features.spadl_converter import (
    ACTION_MAPPING_VERSION,
    COORDINATE_SYSTEM_VERSION,
)
from matchmind.analytics.features.vaep_features import (
    BASE_FEATURE_VERSION,
    NB_PREV_ACTIONS,
)


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


class FeatureArtifactValidationError(ValueError):
    """Raised when a feature artifact cannot safely enter model training."""


@dataclass(frozen=True, slots=True)
class FeatureArtifactLineage:
    analytics_run_id: int | None
    dataset_fingerprint: str
    mapping_version: str
    coordinate_system_version: str
    state_contract_version: str
    feature_version: str


@dataclass(frozen=True, slots=True)
class FeatureArtifacts:
    directory: Path
    actions: Any
    features: Any
    manifest: dict[str, Any]
    quality_report: dict[str, Any]
    lineage: FeatureArtifactLineage
    feature_allowlist: tuple[str, ...]


@lru_cache(maxsize=1)
def baseline_feature_allowlist() -> tuple[str, ...]:
    """Return socceraction's exact default a0/a1/a2 model feature contract."""

    if NB_PREV_ACTIONS != 3:
        raise FeatureArtifactValidationError(
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
        raise FeatureArtifactValidationError("baseline feature allowlist is not unique")
    if any(name.endswith("_a3") for name in names):
        raise FeatureArtifactValidationError("baseline feature allowlist contains a3")
    return names


class FeatureArtifactLoader:
    """Load feature-artifact Parquet only after all lineage contracts reconcile."""

    def load(
        self,
        artifact_directory: Path,
        *,
        expected_dataset_fingerprint: str | None = None,
        expected_analytics_run_id: int | None = None,
    ) -> FeatureArtifacts:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "feature artifact loading requires pyarrow; install requirements.txt"
            ) from exc

        directory = Path(artifact_directory).resolve()
        if not directory.is_dir():
            raise FeatureArtifactValidationError(
                f"feature artifact directory does not exist: {directory}"
            )

        manifest = self._read_json(directory / MANIFEST_FILENAME, "manifest")
        declared_paths = self._verify_hashes(directory, manifest)
        quality_report = self._read_json(
            declared_paths[QUALITY_REPORT_FILENAME], "quality report"
        )
        lineage = self._validate_lineage(
            directory,
            manifest,
            quality_report,
            expected_dataset_fingerprint=expected_dataset_fingerprint,
            expected_analytics_run_id=expected_analytics_run_id,
        )

        action_path = declared_paths[ACTION_FILENAME]
        feature_path = declared_paths[FEATURE_FILENAME]
        action_schema = pq.read_schema(action_path)
        feature_schema = pq.read_schema(feature_path)
        self._validate_schemas(action_schema, feature_schema, lineage)

        actions = pq.read_table(action_path)
        features = pq.read_table(feature_path)
        self._validate_rows(actions, features, quality_report, lineage)
        self._verify_dataset_fingerprint(actions, quality_report, lineage)

        return FeatureArtifacts(
            directory=directory,
            actions=actions,
            features=features,
            manifest=manifest,
            quality_report=quality_report,
            lineage=lineage,
            feature_allowlist=baseline_feature_allowlist(),
        )

    def _verify_hashes(
        self, directory: Path, manifest: dict[str, Any]
    ) -> dict[str, Path]:
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise FeatureArtifactValidationError("manifest files must be an object")
        required = {ACTION_FILENAME, FEATURE_FILENAME, QUALITY_REPORT_FILENAME}
        missing = required - set(files)
        if missing:
            raise FeatureArtifactValidationError(
                f"manifest is missing files: {sorted(missing)}"
            )

        paths: dict[str, Path] = {}
        for filename, expected_hash in files.items():
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise FeatureArtifactValidationError(
                    f"manifest contains unsafe file name: {filename!r}"
                )
            if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(
                expected_hash
            ):
                raise FeatureArtifactValidationError(
                    f"manifest has invalid SHA-256 for {filename}"
                )
            path = directory / filename
            if not path.is_file():
                raise FeatureArtifactValidationError(f"artifact file is missing: {path}")
            actual_hash = self._sha256(path)
            if actual_hash != expected_hash:
                raise FeatureArtifactValidationError(
                    f"SHA-256 mismatch for {filename}: expected {expected_hash}, "
                    f"got {actual_hash}"
                )
            paths[filename] = path
        return paths

    @staticmethod
    def _validate_lineage(
        directory: Path,
        manifest: dict[str, Any],
        quality_report: dict[str, Any],
        *,
        expected_dataset_fingerprint: str | None,
        expected_analytics_run_id: int | None,
    ) -> FeatureArtifactLineage:
        expected_versions = {
            "mapping_version": ACTION_MAPPING_VERSION,
            "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
            "state_contract_version": STATE_CONTRACT_VERSION,
            "feature_version": BASE_FEATURE_VERSION,
        }
        for field, expected in expected_versions.items():
            quality_value = quality_report.get(field)
            if quality_value != expected:
                raise FeatureArtifactValidationError(
                    f"quality report {field} must be {expected!r}, got "
                    f"{quality_value!r}"
                )
            if field == "state_contract_version" and field not in manifest:
                continue  # Legacy feature manifests store this in the hashed report.
            manifest_value = manifest.get(field)
            if manifest_value != expected:
                raise FeatureArtifactValidationError(
                    f"manifest {field} must be {expected!r}, got {manifest_value!r}"
                )

        if manifest.get("include_360") is not False:
            raise FeatureArtifactValidationError(
                "baseline modeling requires include_360=false"
            )
        if quality_report.get("include_360") is not False:
            raise FeatureArtifactValidationError(
                "quality report does not describe a baseline feature artifact"
            )

        fingerprint = manifest.get("dataset_fingerprint")
        if not isinstance(fingerprint, str) or not _SHA256_PATTERN.fullmatch(
            fingerprint
        ):
            raise FeatureArtifactValidationError(
                "manifest dataset_fingerprint must be a lowercase SHA-256"
            )
        if (
            expected_dataset_fingerprint is not None
            and fingerprint != expected_dataset_fingerprint
        ):
            raise FeatureArtifactValidationError(
                "dataset fingerprint does not match the requested lineage"
            )
        if directory.name.startswith("dataset-") and directory.name != (
            f"dataset-{fingerprint[:16]}"
        ):
            raise FeatureArtifactValidationError(
                "artifact directory does not match dataset fingerprint"
            )

        analytics_run_id = manifest.get("analytics_run_id")
        if analytics_run_id is not None and (
            isinstance(analytics_run_id, bool)
            or not isinstance(analytics_run_id, int)
            or analytics_run_id <= 0
        ):
            raise FeatureArtifactValidationError(
                "manifest analytics_run_id must be a positive integer"
            )
        if (
            expected_analytics_run_id is not None
            and analytics_run_id != expected_analytics_run_id
        ):
            raise FeatureArtifactValidationError(
                "analytics run does not match the requested lineage"
            )

        return FeatureArtifactLineage(
            analytics_run_id=analytics_run_id,
            dataset_fingerprint=fingerprint,
            mapping_version=ACTION_MAPPING_VERSION,
            coordinate_system_version=COORDINATE_SYSTEM_VERSION,
            state_contract_version=STATE_CONTRACT_VERSION,
            feature_version=BASE_FEATURE_VERSION,
        )

    @staticmethod
    def _validate_schemas(
        action_schema: Any, feature_schema: Any, lineage: FeatureArtifactLineage
    ) -> None:
        missing_action_columns = REQUIRED_ACTION_COLUMNS - set(action_schema.names)
        if missing_action_columns:
            raise FeatureArtifactValidationError(
                f"actions schema is missing columns: {sorted(missing_action_columns)}"
            )

        allowlist = baseline_feature_allowlist()
        expected_feature_columns = FEATURE_METADATA_COLUMNS + allowlist
        if tuple(feature_schema.names) != expected_feature_columns:
            actual = set(feature_schema.names)
            expected = set(expected_feature_columns)
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            order_mismatch = not missing and not unexpected
            raise FeatureArtifactValidationError(
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
                    raise FeatureArtifactValidationError(
                        f"{label} metadata {field} must be {expected!r}"
                    )
            optional_metadata = {
                "state_contract_version": lineage.state_contract_version,
                "dataset_fingerprint": lineage.dataset_fingerprint,
            }
            for field, expected in optional_metadata.items():
                if field in metadata and metadata[field] != expected:
                    raise FeatureArtifactValidationError(
                        f"{label} metadata {field} must be {expected!r}"
                    )

    @staticmethod
    def _validate_rows(
        actions: Any,
        features: Any,
        quality_report: dict[str, Any],
        lineage: FeatureArtifactLineage,
    ) -> None:
        action_keys = FeatureArtifactLoader._keys(actions, ACTION_FILENAME)
        feature_keys = FeatureArtifactLoader._keys(features, FEATURE_FILENAME)
        if len(action_keys) != len(feature_keys):
            raise FeatureArtifactValidationError(
                "action and feature row counts do not reconcile"
            )
        if len(set(action_keys)) != len(action_keys):
            raise FeatureArtifactValidationError("actions contain duplicate keys")
        if len(set(feature_keys)) != len(feature_keys):
            raise FeatureArtifactValidationError("features contain duplicate keys")
        if set(action_keys) != set(feature_keys):
            raise FeatureArtifactValidationError(
                "action and feature (match_id, action_id) keys do not reconcile"
            )

        ids_by_match: defaultdict[int, list[int]] = defaultdict(list)
        for match_id, action_id in action_keys:
            ids_by_match[match_id].append(action_id)
        for match_id, action_ids in ids_by_match.items():
            if action_ids != list(range(len(action_ids))):
                raise FeatureArtifactValidationError(
                    f"match {match_id} action_id must be contiguous and zero-based"
                )

        FeatureArtifactLoader._require_single_value(
            actions, "mapping_version", lineage.mapping_version, ACTION_FILENAME
        )
        FeatureArtifactLoader._require_single_value(
            actions,
            "coordinate_system_version",
            lineage.coordinate_system_version,
            ACTION_FILENAME,
        )
        FeatureArtifactLoader._require_single_value(
            actions, "state_version", lineage.state_contract_version, ACTION_FILENAME
        )
        FeatureArtifactLoader._require_single_value(
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
                raise FeatureArtifactValidationError(
                    f"quality report {field} must be {expected}, got "
                    f"{quality_report.get(field)!r}"
                )

    @staticmethod
    def _verify_dataset_fingerprint(
        actions: Any,
        quality_report: dict[str, Any],
        lineage: FeatureArtifactLineage,
    ) -> None:
        event_count = quality_report.get("event_count")
        if isinstance(event_count, bool) or not isinstance(event_count, int):
            raise FeatureArtifactValidationError(
                "quality report event_count must be an integer"
            )
        digest = hashlib.sha256()
        digest.update(lineage.mapping_version.encode())
        digest.update(lineage.feature_version.encode())
        digest.update(str(event_count).encode())
        columns = [
            actions[name].to_pylist()
            for name in (
                "match_id",
                "source",
                "source_event_id",
                "source_event_index",
                "source_action_index",
            )
        ]
        for values in zip(*columns, strict=True):
            digest.update(("|".join(str(value) for value in values) + "\n").encode())
        if digest.hexdigest() != lineage.dataset_fingerprint:
            raise FeatureArtifactValidationError(
                "dataset fingerprint does not reconcile with actions and quality report"
            )

    @staticmethod
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
                raise FeatureArtifactValidationError(
                    f"{label} keys must be non-null integers"
                )
            keys.append((match_id, action_id))
        return keys

    @staticmethod
    def _require_single_value(
        table: Any, column: str, expected: str, label: str
    ) -> None:
        actual = set(table[column].to_pylist())
        if actual != {expected}:
            raise FeatureArtifactValidationError(
                f"{label} column {column} must contain only {expected!r}, got "
                f"{sorted(str(value) for value in actual)}"
            )

    @staticmethod
    def _read_json(path: Path, label: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FeatureArtifactValidationError(
                f"cannot read {label} JSON at {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise FeatureArtifactValidationError(f"{label} must be a JSON object")
        return value

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "ACTION_FILENAME",
    "FEATURE_FILENAME",
    "FEATURE_METADATA_COLUMNS",
    "MANIFEST_FILENAME",
    "FeatureArtifactValidationError",
    "FeatureArtifactLoader",
    "FeatureArtifacts",
    "FeatureArtifactLineage",
    "QUALITY_REPORT_FILENAME",
    "REQUIRED_ACTION_COLUMNS",
    "baseline_feature_allowlist",
]
