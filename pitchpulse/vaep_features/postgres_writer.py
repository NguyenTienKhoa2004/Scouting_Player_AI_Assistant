"""Save generated football actions and features to PostgreSQL."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from .action_state import STATE_CONTRACT_VERSION
from .feature_pipeline import AnalyticsDataset


class PostgresAnalyticsWriter:
    REQUIRED_TABLES = frozenset(
        {
            "meta.analytics_runs",
            "gold.spadl_actions",
            "gold.spadl_action_features",
        }
    )
    ACTION_COLUMNS = (
        "match_id", "action_id", "source", "source_event_id",
        "source_event_index", "source_action_index", "mapping_version",
        "coordinate_system_version", "state_version", "period_id",
        "time_seconds", "minute", "team_id", "player_id", "possession_id",
        "possession_team_id", "type_id", "type_name", "subtype_name",
        "result_id", "result_name", "bodypart_id", "bodypart_name",
        "start_x", "start_y", "end_x", "end_y", "duration", "recipient_id",
        "play_pattern", "under_pressure", "counterpress", "has_360",
        "synthetic", "home_score", "away_score", "score_for",
        "score_against", "goal_difference", "possession_changed",
        "acting_team_has_possession", "is_shootout",
    )

    def __init__(self, connection: Any) -> None:
        self.connection = connection

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
            raise RuntimeError(
                "analytics schema is missing tables: " + ", ".join(sorted(missing))
            )

    def create_run(
        self,
        dataset: AnalyticsDataset,
        match_ids: Sequence[int],
        artifact_directory: Path,
    ) -> int:
        row = self.connection.execute(
            """
            INSERT INTO meta.analytics_runs (
                mapping_version, coordinate_system_version,
                state_contract_version, feature_version, include_360,
                selected_match_ids, artifact_directory
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                dataset.conversion_report.mapping_version,
                dataset.conversion_report.coordinate_system_version,
                STATE_CONTRACT_VERSION,
                dataset.feature_version,
                dataset.include_360,
                sorted(set(match_ids)),
                str(artifact_directory),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("analytics run insert did not return an ID")
        return int(row[0])

    def write_dataset(self, run_id: int, dataset: AnalyticsDataset) -> None:
        if len(dataset.actions) != len(dataset.states):
            raise ValueError("every action must have exactly one state")
        if len(dataset.actions) != len(dataset.features):
            raise ValueError("every action must have exactly one feature row")

        placeholders = ", ".join(["%s"] * (len(self.ACTION_COLUMNS) + 1))
        columns = ", ".join(("run_id", *self.ACTION_COLUMNS))
        action_rows = []
        for state in dataset.states:
            values = state.as_dict()
            action_rows.append(
                (run_id, *(values[column] for column in self.ACTION_COLUMNS))
            )
        self.connection.cursor().executemany(
            f"INSERT INTO gold.spadl_actions ({columns}) VALUES ({placeholders})",
            action_rows,
        )

        self.connection.cursor().executemany(
            """
            INSERT INTO gold.spadl_action_features (
                run_id, match_id, action_id, feature_version,
                include_360, features
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    run_id,
                    row.match_id,
                    row.action_id,
                    row.feature_version,
                    dataset.include_360,
                    Jsonb(dict(row.values)),
                )
                for row in dataset.features
            ],
        )

    def complete_run(self, run_id: int, dataset: AnalyticsDataset) -> None:
        self.connection.execute(
            """
            UPDATE meta.analytics_runs
            SET status = 'succeeded', completed_at = CURRENT_TIMESTAMP,
                event_count = %s, action_count = %s, feature_count = %s,
                quality_report = %s
            WHERE id = %s AND status = 'running'
            """,
            (
                dataset.conversion_report.event_count,
                len(dataset.actions),
                len(dataset.features),
                Jsonb(dataset.quality_report()),
                run_id,
            ),
        )

    def fail_run(self, run_id: int, error_message: str) -> None:
        self.connection.execute(
            """
            UPDATE meta.analytics_runs
            SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
                error_message = %s
            WHERE id = %s AND status = 'running'
            """,
            (error_message, run_id),
        )


__all__ = ["PostgresAnalyticsWriter"]
