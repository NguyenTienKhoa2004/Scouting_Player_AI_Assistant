"""Register a Parquet-first Plan 03 dataset as a real analytics run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from psycopg.types.json import Jsonb

from pitchpulse.shared.file_io import file_sha256, read_json, write_json

from .postgres_writer import PostgresAnalyticsWriter


@dataclass(frozen=True, slots=True)
class AnalyticsRegistrationResult:
    analytics_run_id: int
    created: bool
    action_count: int
    match_count: int


class AnalyticsArtifactRegistrationError(ValueError):
    """Raised when a Plan 03 artifact cannot be registered safely."""


class PostgresAnalyticsArtifactRegistrar:
    """Persist action identities while keeping wide features in Parquet."""

    REQUIRED_TABLES = frozenset(
        {"meta.analytics_runs", "gold.spadl_actions"}
    )

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def register(
        self,
        artifact_directory: Path,
        *,
        batch_size: int = 10_000,
        progress: Callable[[str], None] | None = None,
    ) -> AnalyticsRegistrationResult:
        import pyarrow.parquet as pq

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._require_schema()
        root = Path(artifact_directory).resolve()
        manifest_path = root / "manifest.json"
        quality_path = root / "conversion_quality_report.json"
        manifest = read_json(manifest_path)
        quality = read_json(quality_path)
        actions_path = root / "actions.parquet"
        features_path = root / "action_features.parquet"
        self._verify_file(manifest, actions_path)
        self._verify_file(manifest, features_path)
        self._verify_file(manifest, quality_path)

        actions = pq.ParquetFile(actions_path)
        features = pq.ParquetFile(features_path)
        action_count = int(actions.metadata.num_rows)
        feature_count = int(features.metadata.num_rows)
        if action_count <= 0 or action_count != feature_count:
            raise AnalyticsArtifactRegistrationError(
                "Plan 03 action and feature counts do not reconcile"
            )
        missing = sorted(
            set(PostgresAnalyticsWriter.ACTION_COLUMNS)
            - set(actions.schema_arrow.names)
        )
        if missing:
            raise AnalyticsArtifactRegistrationError(
                f"actions.parquet lacks columns: {missing}"
            )
        match_ids = sorted(
            {
                int(value)
                for batch in actions.iter_batches(columns=["match_id"])
                for value in batch.column(0).to_pylist()
            }
        )
        if len(match_ids) != int(quality.get("match_count", -1)):
            raise AnalyticsArtifactRegistrationError(
                "Plan 03 match count does not reconcile"
            )
        if action_count != int(quality.get("action_count", -1)):
            raise AnalyticsArtifactRegistrationError(
                "Plan 03 quality report has a different action count"
            )

        existing = self._find_existing(root, manifest, match_ids, action_count)
        if existing is not None:
            self._record_run_id(manifest_path, manifest, existing)
            return AnalyticsRegistrationResult(
                analytics_run_id=existing,
                created=False,
                action_count=action_count,
                match_count=len(match_ids),
            )

        quality_report = dict(quality)
        quality_report.update(
            {
                "dataset_fingerprint": manifest.get("dataset_fingerprint"),
                "feature_storage": "parquet",
                "database_action_storage": "gold.spadl_actions",
                "database_feature_storage": "not_materialized",
            }
        )
        with self.connection.transaction():
            row = self.connection.execute(
                """
                INSERT INTO meta.analytics_runs (
                    mapping_version, coordinate_system_version,
                    state_contract_version, feature_version, include_360,
                    selected_match_ids, artifact_directory
                ) VALUES (%s, %s, %s, %s, %s, %s::bigint[], %s)
                RETURNING id
                """,
                (
                    manifest.get("mapping_version"),
                    manifest.get("coordinate_system_version"),
                    manifest.get("state_contract_version"),
                    manifest.get("feature_version"),
                    bool(manifest.get("include_360")),
                    match_ids,
                    str(root),
                ),
            ).fetchone()
            if row is None:
                raise AnalyticsArtifactRegistrationError(
                    "Analytics run insert did not return an ID"
                )
            run_id = int(row[0])
            columns = PostgresAnalyticsWriter.ACTION_COLUMNS
            copy_sql = (
                "COPY gold.spadl_actions (run_id, "
                + ", ".join(columns)
                + ") FROM STDIN"
            )
            copied = 0
            with self.connection.cursor().copy(copy_sql) as copy:
                for batch in actions.iter_batches(
                    batch_size=batch_size, columns=list(columns)
                ):
                    for action in batch.to_pylist():
                        copy.write_row((run_id, *(action[name] for name in columns)))
                        copied += 1
            if copied != action_count:
                raise AnalyticsArtifactRegistrationError(
                    "Copied Plan 03 action count does not reconcile"
                )
            self.connection.execute(
                """
                UPDATE meta.analytics_runs
                SET status = 'succeeded', completed_at = CURRENT_TIMESTAMP,
                    event_count = %s, action_count = %s, feature_count = %s,
                    quality_report = %s
                WHERE id = %s AND status = 'running'
                """,
                (
                    int(quality.get("event_count", 0)),
                    action_count,
                    feature_count,
                    Jsonb(quality_report),
                    run_id,
                ),
            )
        self._record_run_id(manifest_path, manifest, run_id)
        if progress is not None:
            progress(
                f"registered {action_count:,} actions from {len(match_ids):,} matches"
            )
        return AnalyticsRegistrationResult(
            analytics_run_id=run_id,
            created=True,
            action_count=action_count,
            match_count=len(match_ids),
        )

    def _require_schema(self) -> None:
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
            raise AnalyticsArtifactRegistrationError(
                "Plan 03 schema is missing: " + ", ".join(sorted(missing))
            )

    def _find_existing(
        self,
        root: Path,
        manifest: dict[str, Any],
        match_ids: list[int],
        action_count: int,
    ) -> int | None:
        row = self.connection.execute(
            """
            SELECT ar.id,
                   (SELECT count(*) FROM gold.spadl_actions aa
                    WHERE aa.run_id = ar.id)
            FROM meta.analytics_runs ar
            WHERE ar.status = 'succeeded'
              AND ar.mapping_version = %s
              AND ar.coordinate_system_version = %s
              AND ar.state_contract_version = %s
              AND ar.feature_version = %s
              AND ar.include_360 = %s
              AND ar.selected_match_ids = %s::bigint[]
              AND ar.artifact_directory = %s
              AND ar.quality_report ->> 'dataset_fingerprint' = %s
            ORDER BY ar.id DESC
            LIMIT 1
            """,
            (
                manifest.get("mapping_version"),
                manifest.get("coordinate_system_version"),
                manifest.get("state_contract_version"),
                manifest.get("feature_version"),
                bool(manifest.get("include_360")),
                match_ids,
                str(root),
                manifest.get("dataset_fingerprint"),
            ),
        ).fetchone()
        if row is None:
            return None
        if int(row[1]) != action_count:
            raise AnalyticsArtifactRegistrationError(
                "Existing analytics run has an incomplete action table"
            )
        return int(row[0])

    @staticmethod
    def _verify_file(manifest: dict[str, Any], path: Path) -> None:
        expected = (manifest.get("files") or {}).get(path.name)
        if not path.is_file() or not isinstance(expected, str):
            raise AnalyticsArtifactRegistrationError(
                f"Plan 03 manifest does not declare {path.name}"
            )
        if file_sha256(path) != expected:
            raise AnalyticsArtifactRegistrationError(
                f"Plan 03 artifact hash mismatch: {path.name}"
            )

    @staticmethod
    def _record_run_id(
        manifest_path: Path, manifest: dict[str, Any], run_id: int
    ) -> None:
        declared = manifest.get("analytics_run_id")
        if declared is not None and int(declared) != run_id:
            raise AnalyticsArtifactRegistrationError(
                "Plan 03 manifest already names a different analytics run"
            )
        updated = dict(manifest)
        updated["analytics_run_id"] = run_id
        write_json(manifest_path, updated)


__all__ = [
    "AnalyticsArtifactRegistrationError",
    "AnalyticsRegistrationResult",
    "PostgresAnalyticsArtifactRegistrar",
]
