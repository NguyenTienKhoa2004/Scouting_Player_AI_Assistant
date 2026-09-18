from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from datetime import time
from pathlib import Path
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from matchmind.spadl import (  # noqa: E402
    SpadlInputContractError,
    validate_spadl_input,
)
from matchmind.spadl.input_reader import (  # noqa: E402
    PostgresSpadlInputReader,
    SPADL_INPUT_SCHEMA,
)
from matchmind.ingestion.normalizer import CanonicalEvent  # noqa: E402
from matchmind.ingestion.lineup_normalizer import (  # noqa: E402
    CanonicalLineupInterval,
)
from matchmind.ingestion.three_sixty_normalizer import (  # noqa: E402
    Canonical360Frame,
)


class SpadlInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = CanonicalEvent(
            source="statsbomb",
            source_event_id=UUID("13b63722-8099-4cdf-818a-d2df3036a633"),
            source_record_index=0,
            source_event_index=1,
            match_id=3857276,
            team_id=788,
            player_id=5237,
            possession_id=1,
            possession_team_id=788,
            event_type="pass",
            event_subtype="kick_off",
            period=1,
            timestamp=time(0, 0, 1),
            minute=0,
            second=1,
            duration=0.5,
            x=50.0,
            y=50.0,
            end_x=60.0,
            end_y=50.0,
            outcome="complete",
            body_part="right_foot",
            recipient_id=6301,
            xg=None,
            play_pattern="from_kick_off",
            under_pressure=False,
            counterpress=False,
            related_event_ids=(),
            raw_details={"pass": {"type": {"name": "Kick Off"}}},
        )
        self.interval = CanonicalLineupInterval(
            source="statsbomb",
            source_record_index=0,
            match_id=3857276,
            team_id=788,
            player_id=5237,
            position_id=21,
            position="Left Wing",
            from_seconds=0,
            to_seconds=5400,
            from_period=1,
            to_period=2,
            start_reason="Starting XI",
            end_reason="Final Whistle",
            raw_details={},
        )
        self.recipient_interval = replace(self.interval, player_id=6301)
        self.frame = Canonical360Frame(
            source="statsbomb",
            source_event_id=self.event.source_event_id,
            source_record_index=0,
            match_id=3857276,
            visible_area=(0.0, 0.0, 120.0, 0.0, 120.0, 80.0),
            freeze_frame=(),
            raw_details={},
        )
    def test_accepts_complete_spadl_input_contract(self) -> None:
        report = validate_spadl_input(
            [self.event], [self.interval, self.recipient_interval], [self.frame]
        )

        self.assertTrue(report.is_valid)
        self.assertEqual(report.event_count, 1)
        self.assertEqual(report.events_with_subtype, 1)
        self.assertTrue(report.as_dict()["valid"])

    def test_rejects_inconsistent_possession_team(self) -> None:
        second = replace(
            self.event,
            source_event_id=UUID("23b63722-8099-4cdf-818a-d2df3036a633"),
            source_record_index=1,
            source_event_index=2,
            possession_team_id=1833,
        )

        report = validate_spadl_input(
            [self.event, second], [self.interval, self.recipient_interval], []
        )

        self.assertIn("possession_team_inconsistent", self._codes(report))

    def test_allows_late_ball_receipt_from_prior_possession(self) -> None:
        current = replace(self.event, possession_id=2)
        late_receipt = replace(
            self.event,
            source_event_id=UUID("23b63722-8099-4cdf-818a-d2df3036a633"),
            source_event_index=2,
            possession_id=1,
            event_type="ball_receipt",
            event_subtype=None,
            end_x=None,
            end_y=None,
            outcome="complete",
            recipient_id=None,
            body_part=None,
            raw_details={"ball_receipt": {}},
        )

        report = validate_spadl_input(
            [current, late_receipt],
            [self.interval, self.recipient_interval],
            [],
        )

        self.assertNotIn("possession_order_invalid", self._codes(report))

    def test_non_action_can_announce_next_possession_before_current_action(self) -> None:
        first = replace(self.event, possession_id=1)
        referee_ball_drop = replace(
            self.event,
            source_event_id=UUID("23b63722-8099-4cdf-818a-d2df3036a633"),
            source_event_index=2,
            possession_id=2,
            event_type="referee_ball_drop",
            event_subtype=None,
            end_x=None,
            end_y=None,
            outcome=None,
            recipient_id=None,
            body_part=None,
            raw_details={},
        )
        late_carry = replace(
            self.event,
            source_event_id=UUID("33b63722-8099-4cdf-818a-d2df3036a633"),
            source_event_index=3,
            possession_id=1,
            event_type="carry",
            event_subtype=None,
            recipient_id=None,
            body_part=None,
            raw_details={"carry": {}},
        )
        next_possession = replace(
            self.event,
            source_event_id=UUID("43b63722-8099-4cdf-818a-d2df3036a633"),
            source_event_index=4,
            possession_id=2,
        )

        report = validate_spadl_input(
            [first, referee_ball_drop, late_carry, next_possession],
            [self.interval, self.recipient_interval],
            [],
        )

        self.assertNotIn("possession_order_invalid", self._codes(report))

    def test_possession_order_uses_period_and_time_like_socceraction(self) -> None:
        period_two = replace(
            self.event,
            source_event_index=1,
            period=2,
            possession_id=204,
        )
        late_source_insertion = replace(
            self.event,
            source_event_id=UUID("23b63722-8099-4cdf-818a-d2df3036a633"),
            source_event_index=2,
            period=1,
            possession_id=97,
        )
        next_period_two = replace(
            self.event,
            source_event_id=UUID("33b63722-8099-4cdf-818a-d2df3036a633"),
            source_event_index=3,
            period=2,
            timestamp=time(0, 0, 2),
            possession_id=205,
        )

        report = validate_spadl_input(
            [period_two, late_source_insertion, next_period_two],
            [self.interval, self.recipient_interval],
            [],
        )

        self.assertNotIn("possession_order_invalid", self._codes(report))

    def test_rejects_possession_decrease_in_chronological_action_order(self) -> None:
        later = replace(
            self.event,
            source_event_id=UUID("23b63722-8099-4cdf-818a-d2df3036a633"),
            source_event_index=2,
            timestamp=time(0, 0, 2),
            possession_id=1,
        )
        report = validate_spadl_input(
            [replace(self.event, possession_id=2), later],
            [self.interval, self.recipient_interval],
            [],
        )

        self.assertIn("possession_order_invalid", self._codes(report))

    def test_reversed_source_tactical_interval_is_a_warning(self) -> None:
        reversed_interval = replace(
            self.interval,
            from_seconds=7205,
            to_seconds=4897,
            from_period=4,
            to_period=2,
            start_reason="Tactical Shift",
        )

        report = validate_spadl_input(
            [self.event], [reversed_interval, self.recipient_interval], []
        )

        self.assertTrue(report.is_valid)
        self.assertIn(
            "lineup_interval_reversed",
            {warning.code for warning in report.warnings},
        )

    def test_excluded_bad_behaviour_player_does_not_require_lineup(self) -> None:
        event = replace(
            self.event,
            player_id=999999,
            event_type="bad_behaviour",
            event_subtype=None,
            end_x=None,
            end_y=None,
            outcome=None,
            recipient_id=None,
            body_part=None,
            raw_details={},
        )

        report = validate_spadl_input([event], [], [])

        self.assertNotIn("event_player_missing_lineup_interval", self._codes(report))

    def test_rejects_event_player_without_lineup_interval(self) -> None:
        report = validate_spadl_input([self.event], [], [])

        self.assertIn("event_player_missing_lineup_interval", self._codes(report))

    def test_rejects_broken_and_cross_match_three_sixty_links(self) -> None:
        wrong_match = replace(self.frame, match_id=999)
        missing = replace(
            self.frame,
            source_event_id=UUID("43b63722-8099-4cdf-818a-d2df3036a633"),
        )

        report = validate_spadl_input(
            [self.event],
            [self.interval, self.recipient_interval],
            [wrong_match, missing],
        )

        self.assertIn("three_sixty_match_mismatch", self._codes(report))
        self.assertIn("three_sixty_event_not_found", self._codes(report))

    def test_fail_fast_exception_keeps_machine_readable_report(self) -> None:
        report = validate_spadl_input([], [], [])

        with self.assertRaises(SpadlInputContractError) as caught:
            report.raise_for_errors()

        self.assertIs(caught.exception.report, report)
        self.assertIn("events_empty", str(caught.exception))

    def test_postgres_reader_verifies_schema_and_consumes_contract(self) -> None:
        connection = FakeConnection(
            self.event, [self.interval, self.recipient_interval], self.frame
        )

        inputs = PostgresSpadlInputReader(connection).load([3857276])

        self.assertTrue(inputs.report.is_valid)
        self.assertEqual(inputs.events, (self.event,))
        self.assertEqual(inputs.home_team_by_match, {3857276: 788})
        self.assertEqual(inputs.away_team_by_match, {3857276: 1833})
        self.assertEqual(
            inputs.three_sixty_by_event[("statsbomb", self.event.source_event_id)],
            self.frame,
        )

    def test_postgres_reader_stops_on_missing_schema_column(self) -> None:
        connection = FakeConnection(
            self.event,
            [self.interval, self.recipient_interval],
            self.frame,
            omitted_schema_column=("events", "source_event_index"),
        )

        with self.assertRaises(SpadlInputContractError) as caught:
            PostgresSpadlInputReader(connection).load([3857276])

        self.assertIn(
            "schema_column_missing", self._codes(caught.exception.report)
        )

    @staticmethod
    def _codes(report: object) -> set[str]:
        return {issue.code for issue in report.issues}  # type: ignore[attr-defined]

class FakeResult:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class FakeConnection:
    def __init__(
        self,
        event: CanonicalEvent,
        intervals: list[CanonicalLineupInterval],
        frame: Canonical360Frame,
        omitted_schema_column: tuple[str, str] | None = None,
    ) -> None:
        self.event = event
        self.intervals = intervals
        self.frame = frame
        self.omitted_schema_column = omitted_schema_column

    def execute(
        self, sql: str, params: tuple[object, ...] = ()
    ) -> FakeResult:
        del params
        if "information_schema.columns" in sql:
            return FakeResult(
                [
                    (table, column, next(iter(types)))
                    for table, columns in SPADL_INPUT_SCHEMA.items()
                    for column, types in columns.items()
                    if (table, column) != self.omitted_schema_column
                ]
            )
        if " FROM silver.events" in sql:
            return FakeResult(
                [
                    tuple(
                        getattr(self.event, column)
                        for column in PostgresSpadlInputReader.EVENT_COLUMNS
                    )
                ]
            )
        if " FROM silver.matches" in sql:
            return FakeResult([(self.event.match_id, self.event.team_id, 1833)])
        if " FROM silver.player_match_intervals" in sql:
            return FakeResult(
                [
                    tuple(
                        getattr(interval, column)
                        for column in PostgresSpadlInputReader.INTERVAL_COLUMNS
                    )
                    for interval in self.intervals
                ]
            )
        if " FROM silver.event_360" in sql:
            return FakeResult(
                [
                    tuple(
                        getattr(self.frame, column)
                        for column in PostgresSpadlInputReader.THREE_SIXTY_COLUMNS
                    )
                ]
            )
        raise AssertionError(f"unexpected SQL: {sql}")


if __name__ == "__main__":
    unittest.main()
