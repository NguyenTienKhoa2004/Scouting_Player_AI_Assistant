"""PostgreSQL adapter for loading validated SPADL conversion inputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from matchmind.feature_engineering.input import (
    ContractIssue,
    SpadlInput,
    SpadlInputContractError,
    SpadlInputValidationReport,
    SpadlInputValidator,
)
from matchmind.ingestion.normalizer import (
    Canonical360Frame,
    CanonicalEvent,
    CanonicalLineupInterval,
)


SPADL_INPUT_SCHEMA: dict[str, dict[str, frozenset[str]]] = {
    "matches": {
        "id": frozenset({"int8"}),
        "home_team_id": frozenset({"int8"}),
        "away_team_id": frozenset({"int8"}),
    },
    "events": {
        "source": frozenset({"text"}),
        "source_event_id": frozenset({"uuid"}),
        "source_record_index": frozenset({"int4"}),
        "source_event_index": frozenset({"int4"}),
        "match_id": frozenset({"int8"}),
        "team_id": frozenset({"int8"}),
        "player_id": frozenset({"int8"}),
        "possession_id": frozenset({"int8"}),
        "possession_team_id": frozenset({"int8"}),
        "event_type": frozenset({"text"}),
        "event_subtype": frozenset({"text"}),
        "period": frozenset({"int2"}),
        "timestamp": frozenset({"time"}),
        "minute": frozenset({"int2"}),
        "second": frozenset({"int2"}),
        "duration": frozenset({"float8"}),
        "x": frozenset({"float8"}),
        "y": frozenset({"float8"}),
        "end_x": frozenset({"float8"}),
        "end_y": frozenset({"float8"}),
        "outcome": frozenset({"text"}),
        "body_part": frozenset({"text"}),
        "recipient_id": frozenset({"int8"}),
        "xg": frozenset({"float8"}),
        "play_pattern": frozenset({"text"}),
        "under_pressure": frozenset({"bool"}),
        "counterpress": frozenset({"bool"}),
        "related_event_ids": frozenset({"_uuid"}),
        "raw_details": frozenset({"jsonb"}),
    },
    "player_match_intervals": {
        "source": frozenset({"text"}),
        "source_record_index": frozenset({"int4"}),
        "match_id": frozenset({"int8"}),
        "team_id": frozenset({"int8"}),
        "player_id": frozenset({"int8"}),
        "position_id": frozenset({"int4"}),
        "position": frozenset({"text"}),
        "from_seconds": frozenset({"int4"}),
        "to_seconds": frozenset({"int4"}),
        "from_period": frozenset({"int2"}),
        "to_period": frozenset({"int2"}),
        "start_reason": frozenset({"text"}),
        "end_reason": frozenset({"text"}),
        "raw_details": frozenset({"jsonb"}),
    },
    "event_360": {
        "source": frozenset({"text"}),
        "source_event_id": frozenset({"uuid"}),
        "source_record_index": frozenset({"int4"}),
        "match_id": frozenset({"int8"}),
        "visible_area": frozenset({"jsonb"}),
        "freeze_frame": frozenset({"jsonb"}),
        "raw_details": frozenset({"jsonb"}),
    },
}


class PostgresSpadlInputReader:
    """Verify the source schema, load records, then validate SPADL inputs."""

    EVENT_COLUMNS = tuple(SPADL_INPUT_SCHEMA["events"])
    INTERVAL_COLUMNS = tuple(SPADL_INPUT_SCHEMA["player_match_intervals"])
    THREE_SIXTY_COLUMNS = tuple(SPADL_INPUT_SCHEMA["event_360"])

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def validate_schema(self) -> tuple[ContractIssue, ...]:
        rows = self.connection.execute(
            """
            SELECT table_name, column_name, udt_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = ANY(%s::text[])
            """,
            (list(SPADL_INPUT_SCHEMA),),
        ).fetchall()
        actual = {(str(row[0]), str(row[1])): str(row[2]) for row in rows}
        issues: list[ContractIssue] = []
        for table, columns in SPADL_INPUT_SCHEMA.items():
            for column, allowed_types in columns.items():
                actual_type = actual.get((table, column))
                if actual_type is None:
                    issues.append(
                        ContractIssue(
                            "schema_column_missing",
                            table,
                            f"required column {table}.{column} is missing",
                        )
                    )
                elif actual_type not in allowed_types:
                    issues.append(
                        ContractIssue(
                            "schema_column_type_invalid",
                            table,
                            f"{table}.{column} has type {actual_type}; expected "
                            f"one of {sorted(allowed_types)}",
                        )
                    )
        return tuple(issues)

    def load(self, match_ids: Sequence[int] | None = None) -> SpadlInput:
        schema_issues = self.validate_schema()
        if schema_issues:
            report = SpadlInputValidationReport(0, 0, 0, 0, 0, schema_issues)
            raise SpadlInputContractError(report)

        params: tuple[Any, ...] = ()
        where = ""
        if match_ids is not None:
            normalized_ids = sorted(set(match_ids))
            if not normalized_ids:
                report = SpadlInputValidationReport(
                    0,
                    0,
                    0,
                    0,
                    0,
                    (
                        ContractIssue(
                            "match_selection_empty",
                            "matches",
                            "match_ids must contain at least one match",
                        ),
                    ),
                )
                raise SpadlInputContractError(report)
            where = " WHERE match_id = ANY(%s::bigint[])"
            params = (normalized_ids,)

        events = tuple(
            self._event_from_row(row)
            for row in self.connection.execute(
                self._select_sql(
                    "events", self.EVENT_COLUMNS, where, "match_id, source_event_index"
                ),
                params,
            ).fetchall()
        )
        intervals = tuple(
            self._interval_from_row(row)
            for row in self.connection.execute(
                self._select_sql(
                    "player_match_intervals",
                    self.INTERVAL_COLUMNS,
                    where,
                    "match_id, player_id, from_seconds, position_id",
                ),
                params,
            ).fetchall()
        )
        frames = tuple(
            self._frame_from_row(row)
            for row in self.connection.execute(
                self._select_sql(
                    "event_360",
                    self.THREE_SIXTY_COLUMNS,
                    where,
                    "match_id, source_record_index, source_event_id",
                ),
                params,
            ).fetchall()
        )

        report = SpadlInputValidator().validate(events, intervals, frames)
        report.raise_for_errors()

        loaded_match_ids = sorted({event.match_id for event in events})
        match_rows = self.connection.execute(
            """
            SELECT id, home_team_id, away_team_id
            FROM matches
            WHERE id = ANY(%s::bigint[])
            ORDER BY id
            """,
            (loaded_match_ids,),
        ).fetchall()
        home_team_by_match = {int(row[0]): int(row[1]) for row in match_rows}
        away_team_by_match = {int(row[0]): int(row[2]) for row in match_rows}
        missing_matches = set(loaded_match_ids) - set(home_team_by_match)
        if missing_matches:
            raise SpadlInputContractError(
                SpadlInputValidationReport(
                    report.match_count,
                    report.event_count,
                    report.lineup_interval_count,
                    report.three_sixty_count,
                    report.events_with_subtype,
                    tuple(
                        ContractIssue(
                            "match_context_missing",
                            "matches",
                            "home-team context is required for SPADL direction",
                            match_id=match_id,
                        )
                        for match_id in sorted(missing_matches)
                    ),
                )
            )
        return SpadlInput(
            events=events,
            lineup_intervals=intervals,
            three_sixty_by_event={
                (frame.source, frame.source_event_id): frame for frame in frames
            },
            home_team_by_match=home_team_by_match,
            away_team_by_match=away_team_by_match,
            report=report,
        )

    @staticmethod
    def _select_sql(
        table: str, columns: Iterable[str], where: str, order_by: str
    ) -> str:
        # Table and column names come only from constants above.
        return (
            f"SELECT {', '.join(columns)} FROM {table}{where} "
            f"ORDER BY {order_by}"
        )

    @classmethod
    def _event_from_row(cls, row: Sequence[Any]) -> CanonicalEvent:
        value = dict(zip(cls.EVENT_COLUMNS, row, strict=True))
        return CanonicalEvent(
            source=value["source"],
            source_event_id=value["source_event_id"],
            source_record_index=value["source_record_index"],
            source_event_index=value["source_event_index"],
            match_id=value["match_id"],
            team_id=value["team_id"],
            player_id=value["player_id"],
            possession_id=value["possession_id"],
            possession_team_id=value["possession_team_id"],
            event_type=value["event_type"],
            event_subtype=value["event_subtype"],
            period=value["period"],
            timestamp=value["timestamp"],
            minute=value["minute"],
            second=value["second"],
            duration=value["duration"],
            x=value["x"],
            y=value["y"],
            end_x=value["end_x"],
            end_y=value["end_y"],
            outcome=value["outcome"],
            body_part=value["body_part"],
            recipient_id=value["recipient_id"],
            xg=value["xg"],
            play_pattern=value["play_pattern"],
            under_pressure=value["under_pressure"],
            counterpress=value["counterpress"],
            related_event_ids=tuple(value["related_event_ids"] or ()),
            raw_details=value["raw_details"],
        )

    @classmethod
    def _interval_from_row(cls, row: Sequence[Any]) -> CanonicalLineupInterval:
        value = dict(zip(cls.INTERVAL_COLUMNS, row, strict=True))
        return CanonicalLineupInterval(**value)

    @classmethod
    def _frame_from_row(cls, row: Sequence[Any]) -> Canonical360Frame:
        value = dict(zip(cls.THREE_SIXTY_COLUMNS, row, strict=True))
        return Canonical360Frame(
            source=value["source"],
            source_event_id=value["source_event_id"],
            source_record_index=value["source_record_index"],
            match_id=value["match_id"],
            visible_area=tuple(value["visible_area"]),
            freeze_frame=tuple(value["freeze_frame"]),
            raw_details=value["raw_details"],
        )


__all__ = ["PostgresSpadlInputReader", "SPADL_INPUT_SCHEMA"]
