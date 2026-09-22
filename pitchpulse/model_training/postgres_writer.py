"""Transactional PostgreSQL persistence for completed VAEP artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from psycopg.types.json import Jsonb

from pitchpulse.shared.file_io import file_sha256

from .training_files import read_json, verify_training_file, write_json


PERSISTENCE_VERSION = "vaep-postgres-persistence-v1"
PERSISTENCE_MANIFEST_FILENAME = "postgres_persistence_manifest.json"
PERSISTED_INPUT_FILES = (
    "action_labels.parquet",
    "split_assignments.parquet",
    "action_values.parquet",
    "player_vaep.parquet",
    "action_values_manifest.json",
    "player_vaep_manifest.json",
    "feature_allowlist.json",
)


class VaepPersistenceError(ValueError):
    """Raised when artifacts and PostgreSQL lineage do not reconcile."""


@dataclass(frozen=True, slots=True)
class PersistenceResult:
    modeling_run_id: int
    analytics_run_id: int
    created: bool
    label_count: int
    action_value_count: int
    player_aggregate_count: int
    manifest: Path


@dataclass(frozen=True, slots=True)
class _ArtifactBundle:
    root: Path
    training_manifest_path: Path
    training_manifest: dict[str, Any]
    labels: Path
    splits: Path
    action_values: Path
    player_vaep: Path
    action_values_manifest: dict[str, Any]
    player_vaep_manifest: dict[str, Any]
    feature_allowlist: dict[str, Any]
    analytics_run_id: int
    run_key: str
    label_count: int
    action_value_count: int
    player_aggregate_count: int
    split_by_match: dict[int, str]


class PostgresVaepWriter:
    """Copy a completed Parquet-first VAEP run into serving tables."""

    REQUIRED_TABLES = frozenset(
        {
            "meta.analytics_runs",
            "meta.vaep_model_runs",
            "gold.spadl_actions",
            "gold.vaep_action_labels",
            "gold.action_values",
            "gold.player_vaep",
        }
    )

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def persist(
        self,
        artifact_directory: Path,
        *,
        analytics_run_id: int | None = None,
        batch_size: int = 10_000,
        progress: Callable[[str], None] | None = None,
    ) -> PersistenceResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.require_schema()
        bundle = self._load_bundle(
            Path(artifact_directory).resolve(), analytics_run_id
        )
        self._validate_analytics_run(bundle)

        modeling_run_id, should_write = self._claim_run(bundle)
        if should_write:
            try:
                with self.connection.transaction():
                    self._copy_labels(modeling_run_id, bundle, batch_size, progress)
                    self._copy_action_values(
                        modeling_run_id, bundle, batch_size, progress
                    )
                    self._copy_player_vaep(
                        modeling_run_id, bundle, batch_size, progress
                    )
                    self._reconcile_database(modeling_run_id, bundle)
                    self.connection.execute(
                        """
                        UPDATE meta.vaep_model_runs
                        SET status = 'succeeded', completed_at = CURRENT_TIMESTAMP,
                            label_count = %s, eligible_action_count = %s,
                            player_aggregate_count = %s, error_message = NULL
                        WHERE id = %s AND status = 'running'
                        """,
                        (
                            bundle.label_count,
                            bundle.action_value_count,
                            bundle.player_aggregate_count,
                            modeling_run_id,
                        ),
                    )
            except Exception as exc:
                self._mark_failed(modeling_run_id, exc)
                raise
        else:
            self._verify_saved_counts(modeling_run_id, bundle)

        manifest_path = self._write_persistence_manifest(
            modeling_run_id, bundle
        )
        return PersistenceResult(
            modeling_run_id=modeling_run_id,
            analytics_run_id=bundle.analytics_run_id,
            created=should_write,
            label_count=bundle.label_count,
            action_value_count=bundle.action_value_count,
            player_aggregate_count=bundle.player_aggregate_count,
            manifest=manifest_path,
        )

    def require_schema(self) -> None:
        rows = self.connection.execute(
            """
            SELECT table_schema || '.' || table_name
            FROM information_schema.tables
            WHERE (table_schema || '.' || table_name) = ANY(%s::text[])
            """,
            (sorted(self.REQUIRED_TABLES),),
        ).fetchall()
        missing = self.REQUIRED_TABLES - {str(row[0]) for row in rows}
        if missing:
            raise VaepPersistenceError(
                "VAEP schema is missing tables: " + ", ".join(sorted(missing))
            )

    @staticmethod
    def _load_bundle(root: Path, override_run_id: int | None) -> _ArtifactBundle:
        import pyarrow.parquet as pq

        manifest_path = root / "training_manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("status") not in {
            "player_aggregation_complete",
            "postgres_persistence_complete",
        }:
            raise VaepPersistenceError(
                "PostgreSQL persistence requires completed player aggregation"
            )
        promotion = manifest.get("production_promotion") or {}
        if not promotion.get("allowed") or not promotion.get("dataset_adequate"):
            raise VaepPersistenceError(
                "PostgreSQL persistence requires the production dataset gate"
            )

        files = {
            name: verify_training_file(root, manifest, name)
            for name in PERSISTED_INPUT_FILES
        }
        value_manifest = read_json(files["action_values_manifest.json"])
        player_manifest = read_json(files["player_vaep_manifest.json"])
        allowlist = read_json(files["feature_allowlist.json"])

        versions = manifest.get("versions") or {}
        required_versions = (
            "action_mapping", "coordinate_system", "state_contract", "feature",
            "target_policy", "split",
        )
        if any(not isinstance(versions.get(name), str) for name in required_versions):
            raise VaepPersistenceError("Training manifest has incomplete versions")
        value_lineage = value_manifest.get("lineage") or {}
        model_lineage_fields = (
            "score_model_version", "concede_model_version",
            "score_calibration_version", "concede_calibration_version",
        )
        if any(
            not isinstance(value_lineage.get(name), str)
            for name in model_lineage_fields
        ):
            raise VaepPersistenceError("Action-value manifest has incomplete lineage")
        minutes_version = (player_manifest.get("minutes_policy") or {}).get("version")
        if not isinstance(minutes_version, str) or not minutes_version:
            raise VaepPersistenceError("Player manifest lacks its minutes policy")
        if not list(allowlist.get("feature_columns") or ()):
            raise VaepPersistenceError("Feature allowlist is empty")

        source_run_id = (manifest.get("source") or {}).get("analytics_run_id")
        lineage_run_id = (value_manifest.get("lineage") or {}).get(
            "analytics_run_id"
        )
        source = manifest.get("source") or {}
        plan03_directory = source.get("plan03_directory") or source.get(
            "feature_artifact_directory"
        )
        plan03_run_id = None
        if isinstance(plan03_directory, str) and plan03_directory:
            plan03_manifest_path = Path(plan03_directory) / "manifest.json"
            if plan03_manifest_path.is_file():
                plan03_manifest = read_json(plan03_manifest_path)
                if plan03_manifest.get("dataset_fingerprint") != source.get(
                    "dataset_fingerprint"
                ):
                    raise VaepPersistenceError(
                        "Current Plan 03 manifest has a different fingerprint"
                    )
                plan03_run_id = plan03_manifest.get("analytics_run_id")
        declared_ids = {
            int(value)
            for value in (source_run_id, lineage_run_id, plan03_run_id)
            if value is not None
        }
        if len(declared_ids) > 1:
            raise VaepPersistenceError("Artifact analytics run IDs disagree")
        if override_run_id is not None and declared_ids and override_run_id not in declared_ids:
            raise VaepPersistenceError(
                "Requested analytics run differs from artifact lineage"
            )
        selected_run_id = override_run_id or next(iter(declared_ids), None)
        if selected_run_id is None or int(selected_run_id) <= 0:
            raise VaepPersistenceError(
                "Artifact has no analytics_run_id; register the Plan 03 dataset "
                "or provide its real run with --analytics-run-id"
            )

        labels = files["action_labels.parquet"]
        splits = files["split_assignments.parquet"]
        values = files["action_values.parquet"]
        player = files["player_vaep.parquet"]
        PostgresVaepWriter._require_columns(
            labels,
            {
                "analytics_run_id", "match_id", "action_id", "scores",
                "concedes", "eligible", "exclusion_reason",
                "target_policy_version",
            },
        )
        PostgresVaepWriter._require_columns(
            values,
            {
                "analytics_run_id", "match_id", "action_id", "player_id", "team_id",
                "p_score_before", "p_score_after", "p_concede_before",
                "p_concede_after", "offensive_value", "defensive_value",
                "vaep_value", "action_mapping_version",
                "state_contract_version", "feature_version",
                "target_policy_version", "split_version",
                "score_model_version", "concede_model_version",
                "score_calibration_version", "concede_calibration_version",
            },
        )
        PostgresVaepWriter._require_columns(
            player,
            {
                "aggregation_version", "minutes_policy_version",
                "aggregation_level", "match_id", "player_id", "team_id",
                "competition_id", "season_id", "position", "action_type",
                "minutes_played", "match_count", "action_count", "player_vaep",
                "offensive_vaep", "defensive_vaep", "vaep_per_90",
                "minimum_minutes_eligible",
            },
        )

        assignments = pq.read_table(splits, columns=["match_id", "split"])
        match_ids = assignments["match_id"].to_pylist()
        split_values = assignments["split"].to_pylist()
        split_by_match = dict(zip(map(int, match_ids), map(str, split_values)))
        if len(split_by_match) != len(match_ids):
            raise VaepPersistenceError("Split assignments contain duplicate matches")
        if set(split_by_match.values()) - {"train", "validation", "test"}:
            raise VaepPersistenceError("Split assignments contain an invalid split")

        counts = {
            "labels": pq.ParquetFile(labels).metadata.num_rows,
            "values": pq.ParquetFile(values).metadata.num_rows,
            "players": pq.ParquetFile(player).metadata.num_rows,
        }
        if counts["labels"] != int((manifest.get("split") or {}).get("counts", {}).get("action_count", -1)):
            raise VaepPersistenceError("Label count does not match training manifest")
        if counts["values"] != int((manifest.get("action_valuation") or {}).get("row_count", -1)):
            raise VaepPersistenceError("Action-value count does not match manifest")
        if counts["players"] != int((manifest.get("player_aggregation") or {}).get("row_count", -1)):
            raise VaepPersistenceError("Player aggregate count does not match manifest")

        fingerprint = str((manifest.get("source") or {}).get("dataset_fingerprint", ""))
        if not fingerprint or value_manifest.get("dataset_fingerprint") != fingerprint:
            raise VaepPersistenceError("Dataset fingerprints do not reconcile")
        if player_manifest.get("dataset_fingerprint") != fingerprint:
            raise VaepPersistenceError("Player aggregate fingerprint does not reconcile")

        artifact_records = manifest.get("artifacts") or {}
        hashes = {
            name: str(artifact_records[name]["sha256"])
            for name in PERSISTED_INPUT_FILES
        }
        key_payload = {
            "version": PERSISTENCE_VERSION,
            "analytics_run_id": int(selected_run_id),
            "dataset_fingerprint": fingerprint,
            "artifact_hashes": hashes,
        }
        run_key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return _ArtifactBundle(
            root=root,
            training_manifest_path=manifest_path,
            training_manifest=manifest,
            labels=labels,
            splits=splits,
            action_values=values,
            player_vaep=player,
            action_values_manifest=value_manifest,
            player_vaep_manifest=player_manifest,
            feature_allowlist=allowlist,
            analytics_run_id=int(selected_run_id),
            run_key=run_key,
            label_count=int(counts["labels"]),
            action_value_count=int(counts["values"]),
            player_aggregate_count=int(counts["players"]),
            split_by_match=split_by_match,
        )

    @staticmethod
    def _require_columns(path: Path, required: set[str]) -> None:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(required - names)
        if missing:
            raise VaepPersistenceError(f"{path.name} lacks columns: {missing}")

    def _validate_analytics_run(self, bundle: _ArtifactBundle) -> None:
        row = self.connection.execute(
            """
            SELECT status, mapping_version, coordinate_system_version,
                   state_contract_version, feature_version, include_360,
                   selected_match_ids, action_count, feature_count,
                   (SELECT count(*) FROM gold.spadl_actions aa
                    WHERE aa.run_id = ar.id)
            FROM meta.analytics_runs ar
            WHERE id = %s
            """,
            (bundle.analytics_run_id,),
        ).fetchone()
        if row is None:
            raise VaepPersistenceError(
                f"Analytics run {bundle.analytics_run_id} does not exist"
            )
        versions = bundle.training_manifest.get("versions") or {}
        expected = (
            "succeeded",
            versions.get("action_mapping"),
            versions.get("coordinate_system"),
            versions.get("state_contract"),
            versions.get("feature"),
            False,
        )
        if tuple(row[:6]) != expected:
            raise VaepPersistenceError(
                "Analytics run status or version lineage does not match artifacts"
            )
        selected_matches = {int(value) for value in (row[6] or ())}
        expected_matches = set(bundle.split_by_match)
        if selected_matches != expected_matches:
            raise VaepPersistenceError(
                "Analytics run match set does not match the 1,831-match artifact"
            )
        if int(row[7]) != bundle.label_count or int(row[8]) != bundle.label_count:
            raise VaepPersistenceError(
                "Analytics run action/feature counts do not match labels"
            )
        if int(row[9]) != bundle.label_count:
            raise VaepPersistenceError(
                "gold.spadl_actions does not contain every artifact action"
            )

    def _claim_run(self, bundle: _ArtifactBundle) -> tuple[int, bool]:
        manifest = bundle.training_manifest
        versions = manifest.get("versions") or {}
        lineage = bundle.action_values_manifest.get("lineage") or {}
        player_policy = bundle.player_vaep_manifest.get("minutes_policy") or {}
        artifacts = {
            name: record.get("sha256")
            for name, record in (manifest.get("artifacts") or {}).items()
            if isinstance(record, Mapping)
            and name != PERSISTENCE_MANIFEST_FILENAME
        }
        values = (
            bundle.run_key,
            bundle.analytics_run_id,
            (manifest.get("source") or {}).get("dataset_fingerprint"),
            versions.get("action_mapping"),
            versions.get("coordinate_system"),
            versions.get("state_contract"),
            versions.get("feature"),
            versions.get("target_policy"),
            versions.get("split"),
            lineage.get("score_model_version"),
            lineage.get("concede_model_version"),
            lineage.get("score_calibration_version"),
            lineage.get("concede_calibration_version"),
            player_policy.get("version"),
            Jsonb(list(bundle.feature_allowlist.get("feature_columns") or ())),
            Jsonb(dict((manifest.get("models") or {}).get("preprocessing") or {})),
            Jsonb(dict(manifest.get("runtime_dependencies") or {})),
            str(bundle.root),
            Jsonb(artifacts),
            Jsonb(dict(manifest.get("split") or {})),
            Jsonb(
                {
                    "production_promotion": manifest.get("production_promotion"),
                    "test_evaluation": manifest.get("test_evaluation"),
                    "test_metrics": (manifest.get("models") or {}).get("test_metrics"),
                }
            ),
        )
        with self.connection.transaction():
            row = self.connection.execute(
                """
                INSERT INTO meta.vaep_model_runs (
                    run_key, analytics_run_id, dataset_fingerprint,
                    action_mapping_version, coordinate_system_version,
                    state_contract_version, feature_version, target_policy_version,
                    split_version, score_model_version, concede_model_version,
                    score_calibration_version, concede_calibration_version,
                    minutes_policy_version, feature_allowlist, preprocessing,
                    runtime_dependencies, artifact_directory, artifact_hashes,
                    split_statistics, evaluation_metrics
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (run_key) DO NOTHING
                RETURNING id
                """,
                values,
            ).fetchone()
            if row is not None:
                return int(row[0]), True
            existing = self.connection.execute(
                """
                SELECT id, status, analytics_run_id
                FROM meta.vaep_model_runs
                WHERE run_key = %s
                FOR UPDATE
                """,
                (bundle.run_key,),
            ).fetchone()
            if existing is None:
                raise VaepPersistenceError("Could not claim the modeling run")
            run_id, status, analytics_id = int(existing[0]), str(existing[1]), int(existing[2])
            if analytics_id != bundle.analytics_run_id:
                raise VaepPersistenceError("Existing modeling run has different lineage")
            if status == "succeeded":
                return run_id, False
            if status == "running":
                raise VaepPersistenceError(
                    f"Modeling run {run_id} is already being persisted"
                )
            self.connection.execute(
                """
                UPDATE meta.vaep_model_runs
                SET status = 'running', started_at = CURRENT_TIMESTAMP,
                    completed_at = NULL, error_message = NULL
                WHERE id = %s AND status = 'failed'
                """,
                (run_id,),
            )
            return run_id, True

    def _copy_labels(
        self, run_id: int, bundle: _ArtifactBundle, batch_size: int,
        progress: Callable[[str], None] | None,
    ) -> None:
        columns = (
            "analytics_run_id", "match_id", "action_id", "scores", "concedes",
            "eligible", "exclusion_reason", "target_policy_version",
        )
        sql = """
            COPY gold.vaep_action_labels (
                modeling_run_id, analytics_run_id, match_id, action_id,
                scores, concedes, eligible, exclusion_reason, split,
                target_policy_version, split_version
            ) FROM STDIN
        """
        split_version = (bundle.training_manifest.get("versions") or {}).get("split")
        with self.connection.cursor().copy(sql) as copy:
            for row in self._parquet_rows(bundle.labels, columns, batch_size):
                self._require_artifact_run_id(row["analytics_run_id"], bundle)
                if row["target_policy_version"] != (
                    bundle.training_manifest.get("versions") or {}
                ).get("target_policy"):
                    raise VaepPersistenceError(
                        "Action-label target policy does not match the model run"
                    )
                match_id = int(row["match_id"])
                copy.write_row(
                    (
                        run_id, bundle.analytics_run_id, match_id,
                        int(row["action_id"]), row["scores"], row["concedes"],
                        bool(row["eligible"]), row["exclusion_reason"],
                        bundle.split_by_match[match_id],
                        row["target_policy_version"], split_version,
                    )
                )
        if progress is not None:
            progress(f"persisted {bundle.label_count:,} action labels")

    def _copy_action_values(
        self, run_id: int, bundle: _ArtifactBundle, batch_size: int,
        progress: Callable[[str], None] | None,
    ) -> None:
        target_columns = (
            "match_id", "action_id", "player_id", "team_id",
            "p_score_before", "p_score_after", "p_concede_before",
            "p_concede_after", "offensive_value", "defensive_value",
            "vaep_value", "action_mapping_version", "state_contract_version",
            "feature_version", "target_policy_version", "split_version",
            "score_model_version", "concede_model_version",
            "score_calibration_version", "concede_calibration_version",
        )
        source_columns = ("analytics_run_id", *target_columns)
        sql = "COPY gold.action_values (modeling_run_id, analytics_run_id, " + ", ".join(target_columns) + ") FROM STDIN"
        versions = bundle.training_manifest.get("versions") or {}
        lineage = bundle.action_values_manifest.get("lineage") or {}
        expected_lineage = {
            "action_mapping_version": versions.get("action_mapping"),
            "state_contract_version": versions.get("state_contract"),
            "feature_version": versions.get("feature"),
            "target_policy_version": versions.get("target_policy"),
            "split_version": versions.get("split"),
            "score_model_version": lineage.get("score_model_version"),
            "concede_model_version": lineage.get("concede_model_version"),
            "score_calibration_version": lineage.get("score_calibration_version"),
            "concede_calibration_version": lineage.get("concede_calibration_version"),
        }
        with self.connection.cursor().copy(sql) as copy:
            for row in self._parquet_rows(
                bundle.action_values, source_columns, batch_size
            ):
                self._require_artifact_run_id(row["analytics_run_id"], bundle)
                if any(row[name] != value for name, value in expected_lineage.items()):
                    raise VaepPersistenceError(
                        "Action-value version lineage does not match the model run"
                    )
                copy.write_row(
                    (
                        run_id,
                        bundle.analytics_run_id,
                        *(row[name] for name in target_columns),
                    )
                )
        if progress is not None:
            progress(f"persisted {bundle.action_value_count:,} action values")

    def _copy_player_vaep(
        self, run_id: int, bundle: _ArtifactBundle, batch_size: int,
        progress: Callable[[str], None] | None,
    ) -> None:
        columns = (
            "aggregation_version", "minutes_policy_version", "aggregation_level",
            "match_id", "player_id", "team_id", "competition_id", "season_id",
            "position", "action_type", "minutes_played", "match_count",
            "action_count", "player_vaep", "offensive_vaep", "defensive_vaep",
            "vaep_per_90", "minimum_minutes_eligible",
        )
        sql = "COPY gold.player_vaep (modeling_run_id, " + ", ".join(columns) + ") FROM STDIN"
        expected_aggregation = bundle.player_vaep_manifest.get(
            "aggregation_version"
        )
        expected_minutes = (
            bundle.player_vaep_manifest.get("minutes_policy") or {}
        ).get("version")
        with self.connection.cursor().copy(sql) as copy:
            for row in self._parquet_rows(bundle.player_vaep, columns, batch_size):
                if (
                    row["aggregation_version"] != expected_aggregation
                    or row["minutes_policy_version"] != expected_minutes
                ):
                    raise VaepPersistenceError(
                        "Player aggregate version lineage does not match its manifest"
                    )
                copy.write_row((run_id, *(row[name] for name in columns)))
        if progress is not None:
            progress(f"persisted {bundle.player_aggregate_count:,} player aggregates")

    @staticmethod
    def _parquet_rows(path: Path, columns: tuple[str, ...], batch_size: int):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
            yield from batch.to_pylist()

    @staticmethod
    def _require_artifact_run_id(
        artifact_run_id: int | None, bundle: _ArtifactBundle
    ) -> None:
        if artifact_run_id is not None and int(artifact_run_id) != bundle.analytics_run_id:
            raise VaepPersistenceError(
                "Parquet analytics_run_id differs from the selected Plan 03 run"
            )

    def _reconcile_database(self, run_id: int, bundle: _ArtifactBundle) -> None:
        row = self.connection.execute(
            """
            SELECT
                (SELECT count(*) FROM gold.vaep_action_labels WHERE modeling_run_id = %s),
                (SELECT count(*) FROM gold.vaep_action_labels
                 WHERE modeling_run_id = %s AND eligible),
                (SELECT count(*) FROM gold.action_values WHERE modeling_run_id = %s),
                (SELECT count(*) FROM gold.player_vaep WHERE modeling_run_id = %s),
                (SELECT coalesce(sum(vaep_value), 0) FROM gold.action_values
                 WHERE modeling_run_id = %s AND player_id IS NOT NULL),
                (SELECT coalesce(sum(player_vaep), 0) FROM gold.player_vaep
                 WHERE modeling_run_id = %s
                   AND aggregation_level = 'competition_season'
                   AND action_type = 'all')
            """,
            (run_id, run_id, run_id, run_id, run_id, run_id),
        ).fetchone()
        if row is None:
            raise VaepPersistenceError("Could not reconcile persisted rows")
        expected = (
            bundle.label_count,
            bundle.action_value_count,
            bundle.action_value_count,
            bundle.player_aggregate_count,
        )
        if tuple(map(int, row[:4])) != expected:
            raise VaepPersistenceError("Persisted row counts do not reconcile")
        if not math.isclose(float(row[4]), float(row[5]), rel_tol=1e-10, abs_tol=1e-8):
            raise VaepPersistenceError("Persisted player VAEP does not reconcile")

    def _verify_saved_counts(self, run_id: int, bundle: _ArtifactBundle) -> None:
        row = self.connection.execute(
            """
            SELECT label_count, eligible_action_count, player_aggregate_count
            FROM meta.vaep_model_runs WHERE id = %s AND status = 'succeeded'
            """,
            (run_id,),
        ).fetchone()
        if row is None or tuple(map(int, row)) != (
            bundle.label_count,
            bundle.action_value_count,
            bundle.player_aggregate_count,
        ):
            raise VaepPersistenceError("Existing modeling run does not reconcile")

    def _mark_failed(self, run_id: int, error: Exception) -> None:
        self.connection.execute(
            """
            UPDATE meta.vaep_model_runs
            SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
                error_message = %s
            WHERE id = %s AND status = 'running'
            """,
            (str(error)[:4000], run_id),
        )

    def _write_persistence_manifest(
        self, run_id: int, bundle: _ArtifactBundle
    ) -> Path:
        path = bundle.root / PERSISTENCE_MANIFEST_FILENAME
        payload = {
            "schema_version": 1,
            "persistence_version": PERSISTENCE_VERSION,
            "run_key": bundle.run_key,
            "modeling_run_id": run_id,
            "analytics_run_id": bundle.analytics_run_id,
            "dataset_fingerprint": (bundle.training_manifest.get("source") or {}).get(
                "dataset_fingerprint"
            ),
            "status": "succeeded",
            "counts": {
                "labels": bundle.label_count,
                "action_values": bundle.action_value_count,
                "player_aggregates": bundle.player_aggregate_count,
            },
        }
        write_json(path, payload)

        updated = dict(bundle.training_manifest)
        updated["stage"] = "postgres_persistence"
        updated["status"] = "postgres_persistence_complete"
        artifacts = dict(updated.get("artifacts") or {})
        artifacts[path.name] = {"path": str(path), "sha256": file_sha256(path)}
        updated["artifacts"] = artifacts
        updated["database_persistence"] = payload
        write_json(bundle.training_manifest_path, updated)
        return path


__all__ = [
    "PERSISTENCE_MANIFEST_FILENAME",
    "PERSISTENCE_VERSION",
    "PERSISTED_INPUT_FILES",
    "PersistenceResult",
    "PostgresVaepWriter",
    "VaepPersistenceError",
]
