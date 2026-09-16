from __future__ import annotations

import sys
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import time
from pathlib import Path
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.analytics.features import (  # noqa: E402
    BASE_FEATURE_VERSION,
    FEATURE_360_VERSION,
    AnalyticsDatasetBuilder,
    SpadlInput,
    SpadlInputValidationReport,
    STATE_CONTRACT_VERSION,
)
from matchmind.data.ingestion.normalizer import Canonical360Frame, CanonicalEvent  # noqa: E402
from matchmind.ml.feature_artifacts import FeatureArtifactLoader  # noqa: E402
from matchmind.data.storage.parquet import (  # noqa: E402
    ChunkedParquetArtifactWriter,
    ParquetArtifactWriter,
)
from matchmind.data.storage.postgres.analytics_writer import (  # noqa: E402
    PostgresAnalyticsWriter,
)


class FeatureArtifactHandCalculatedFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pass_event = self._event(
            event_id="10000000-0000-0000-0000-000000000001",
            index=1,
            team_id=10,
            player_id=101,
            possession_id=1,
            possession_team_id=10,
            event_type="pass",
            timestamp=time(0, 0, 1),
            x=10.0,
            y=50.0,
            end_x=20.0,
            end_y=50.0,
            outcome="complete",
            raw_details={"pass": {}},
        )
        self.goal = self._event(
            event_id="10000000-0000-0000-0000-000000000002",
            index=2,
            team_id=10,
            player_id=102,
            possession_id=1,
            possession_team_id=10,
            event_type="shot",
            timestamp=time(0, 0, 2),
            x=90.0,
            y=50.0,
            end_x=100.0,
            end_y=50.0,
            outcome="goal",
            raw_details={"shot": {}},
        )
        self.away_pass = self._event(
            event_id="10000000-0000-0000-0000-000000000003",
            index=3,
            team_id=20,
            player_id=201,
            possession_id=2,
            possession_team_id=20,
            event_type="pass",
            timestamp=time(0, 0, 3),
            x=50.0,
            y=50.0,
            end_x=40.0,
            end_y=50.0,
            outcome="complete",
            raw_details={"pass": {}},
        )
        self.own_goal = self._event(
            event_id="10000000-0000-0000-0000-000000000004",
            index=4,
            team_id=20,
            player_id=202,
            possession_id=2,
            possession_team_id=20,
            event_type="own_goal_against",
            timestamp=time(0, 0, 20),
            x=5.0,
            y=50.0,
            end_x=None,
            end_y=None,
            outcome=None,
            raw_details={},
        )

    def test_score_and_possession_state_are_after_each_action(self) -> None:
        dataset = AnalyticsDatasetBuilder().build(
            self._inputs(
                [self.pass_event, self.goal, self.away_pass, self.own_goal]
            )
        )

        self.assertEqual(
            [(state.home_score, state.away_score) for state in dataset.states],
            [(0, 0), (1, 0), (1, 0), (2, 0)],
        )
        self.assertEqual(
            [state.goal_difference for state in dataset.states], [0, 1, -1, -2]
        )
        self.assertEqual(
            [state.possession_changed for state in dataset.states],
            [False, False, True, False],
        )

    def test_shootout_goal_does_not_change_match_score(self) -> None:
        shootout_goal = replace(
            self.goal,
            period=5,
            minute=120,
            timestamp=time(0, 0, 1),
        )

        state = AnalyticsDatasetBuilder().build(
            self._inputs([shootout_goal])
        ).states[0]

        self.assertTrue(state.is_shootout)
        self.assertEqual((state.home_score, state.away_score), (0, 0))

    def test_standard_vaep_state_contains_exactly_three_actions(self) -> None:
        dataset = AnalyticsDatasetBuilder().build(
            self._inputs(
                [self.pass_event, self.goal, self.away_pass, self.own_goal]
            )
        )
        row = dataset.features[2].as_dict()

        self.assertEqual(row["feature_version"], BASE_FEATURE_VERSION)
        self.assertTrue(row["actiontype_pass_a0"])
        self.assertTrue(row["actiontype_shot_a1"])
        self.assertTrue(row["actiontype_pass_a2"])
        self.assertFalse(any(name.endswith("_a3") for name in row))
        self.assertEqual(row["time_delta_1"], 1.0)
        self.assertFalse(row["team_1"])
        self.assertEqual(row["goalscore_diff"], -1)

    def test_future_action_cannot_change_an_existing_feature_row(self) -> None:
        before = AnalyticsDatasetBuilder().build(
            self._inputs([self.pass_event, self.goal, self.away_pass])
        )
        after = AnalyticsDatasetBuilder().build(
            self._inputs(
                [self.pass_event, self.goal, self.away_pass, self.own_goal]
            )
        )

        self.assertEqual(before.features[2], after.features[2])

    def test_history_does_not_cross_match_boundaries(self) -> None:
        other = replace(
            self.pass_event,
            source_event_id=UUID("20000000-0000-0000-0000-000000000001"),
            match_id=200,
            team_id=30,
            player_id=301,
        )
        dataset = AnalyticsDatasetBuilder().build(
            self._inputs(
                [self.pass_event, other],
                home={100: 10, 200: 30},
                away={100: 20, 200: 40},
            )
        )

        second_match_row = dataset.features[1].as_dict()
        self.assertEqual(second_match_row["time_delta_1"], 0.0)
        self.assertTrue(second_match_row["team_1"])

    def test_360_has_an_explicit_version_and_missing_values(self) -> None:
        frame = Canonical360Frame(
            source="statsbomb",
            source_event_id=self.goal.source_event_id,
            source_record_index=0,
            match_id=100,
            visible_area=(0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0),
            freeze_frame=(
                {"actor": True, "teammate": True, "location": [108.0, 40.0]},
                {"actor": False, "teammate": True, "location": [100.0, 40.0]},
                {"actor": False, "teammate": False, "location": [109.0, 40.0]},
            ),
            raw_details={},
        )
        inputs = self._inputs([self.pass_event, self.goal], frames=[frame])

        baseline = AnalyticsDatasetBuilder().build(inputs)
        enriched = AnalyticsDatasetBuilder().build(inputs, include_360=True)

        self.assertNotIn("nearest_defender_distance", baseline.features[1].as_dict())
        goal_row = enriched.features[1].as_dict()
        self.assertEqual(goal_row["feature_version"], FEATURE_360_VERSION)
        self.assertTrue(goal_row["has_360"])
        self.assertEqual(goal_row["visible_teammates"], 1)
        self.assertEqual(goal_row["visible_opponents"], 1)
        self.assertEqual(goal_row["pressure_around_ball"], 1)
        self.assertTrue(goal_row["goal_visible"])
        self.assertIsNotNone(goal_row["visible_goal_angle"])
        missing_row = enriched.features[0].as_dict()
        self.assertFalse(missing_row["has_360"])
        self.assertIsNone(missing_row["nearest_defender_distance"])

    def test_writes_versioned_reproducible_parquet_artifacts(self) -> None:
        import pyarrow.parquet as pq

        dataset = AnalyticsDatasetBuilder().build(
            self._inputs([self.pass_event, self.goal])
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = ParquetArtifactWriter().write(dataset, Path(directory))
            first_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            second_paths = ParquetArtifactWriter().write(dataset, Path(directory))
            second_manifest = json.loads(
                second_paths.manifest.read_text(encoding="utf-8")
            )

            self.assertEqual(pq.read_table(paths.actions).num_rows, 2)
            self.assertEqual(pq.read_table(paths.features).num_rows, 2)
            self.assertEqual(first_manifest, second_manifest)
            self.assertIn(BASE_FEATURE_VERSION, str(paths.directory))
            self.assertEqual(
                first_manifest["state_contract_version"], STATE_CONTRACT_VERSION
            )
            action_metadata = pq.read_schema(paths.actions).metadata or {}
            self.assertEqual(
                action_metadata[b"state_contract_version"].decode(),
                STATE_CONTRACT_VERSION,
            )
            self.assertEqual(
                action_metadata[b"dataset_fingerprint"].decode(),
                first_manifest["dataset_fingerprint"],
            )

            other_dataset = AnalyticsDatasetBuilder().build(
                self._inputs([self.pass_event])
            )
            other_paths = ParquetArtifactWriter().write(
                other_dataset, Path(directory)
            )
            self.assertNotEqual(paths.directory, other_paths.directory)

    def test_writes_one_valid_artifact_from_bounded_batches(self) -> None:
        first = AnalyticsDatasetBuilder().build(
            self._inputs([self.pass_event, self.goal])
        )
        second_events = [
            replace(
                self.pass_event,
                match_id=200,
                source_event_id=UUID(
                    "20000000-0000-0000-0000-000000000001"
                ),
            ),
            replace(
                self.goal,
                match_id=200,
                source_event_id=UUID(
                    "20000000-0000-0000-0000-000000000002"
                ),
            ),
        ]
        second = AnalyticsDatasetBuilder().build(
            self._inputs(second_events, home={200: 10}, away={200: 20})
        )
        combined = AnalyticsDatasetBuilder().build(
            self._inputs(
                [self.pass_event, self.goal, *second_events],
                home={100: 10, 200: 10},
                away={100: 20, 200: 20},
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = ChunkedParquetArtifactWriter().write(
                [first, second], Path(directory)
            )
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            quality = json.loads(
                paths.quality_report.read_text(encoding="utf-8")
            )
            loaded = FeatureArtifactLoader().load(paths.directory)

            self.assertEqual(quality["match_count"], 2)
            self.assertEqual(loaded.actions.num_rows, 4)
            self.assertEqual(loaded.features.num_rows, 4)
            self.assertEqual(
                manifest["dataset_fingerprint"],
                ParquetArtifactWriter._dataset_fingerprint(combined),
            )

    def test_materializes_one_action_and_feature_per_state(self) -> None:
        dataset = AnalyticsDatasetBuilder().build(
            self._inputs([self.pass_event, self.goal])
        )
        connection = FakeAnalyticsConnection()
        writer = PostgresAnalyticsWriter(connection)

        writer.require_schema()
        run_id = writer.create_run(dataset, [100], Path("artifacts/test"))
        writer.write_dataset(run_id, dataset)
        writer.complete_run(run_id, dataset)

        self.assertEqual(run_id, 7)
        self.assertEqual(len(connection.batch_rows[0]), 2)
        self.assertEqual(len(connection.batch_rows[1]), 2)

    @staticmethod
    def _event(
        *,
        event_id: str,
        index: int,
        team_id: int,
        player_id: int,
        possession_id: int,
        possession_team_id: int,
        event_type: str,
        timestamp: time,
        x: float,
        y: float,
        end_x: float | None,
        end_y: float | None,
        outcome: str | None,
        raw_details: dict[str, object],
    ) -> CanonicalEvent:
        source_type = {
            "pass": "Pass",
            "shot": "Shot",
            "own_goal_against": "Own Goal Against",
        }[event_type]
        payload: dict[str, object] = {
            "id": event_id,
            "index": index,
            "period": 1,
            "timestamp": timestamp.isoformat(),
            "minute": 0,
            "second": timestamp.second,
            "possession": possession_id,
            "possession_team": {
                "id": possession_team_id,
                "name": "Possession team",
            },
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": team_id, "name": "Team"},
            "player": {"id": player_id, "name": "Player"},
            "type": {"id": 30, "name": source_type},
            "location": [x / 100.0 * 120.0, y / 100.0 * 80.0],
            **raw_details,
        }
        if event_type in {"pass", "shot"}:
            detail = dict(payload[event_type])  # type: ignore[arg-type]
            if end_x is not None and end_y is not None:
                detail["end_location"] = [
                    end_x / 100.0 * 120.0,
                    end_y / 100.0 * 80.0,
                ]
            detail["body_part"] = {"id": 40, "name": "Right Foot"}
            if event_type == "shot" and outcome == "goal":
                detail["outcome"] = {"id": 97, "name": "Goal"}
            payload[event_type] = detail

        return CanonicalEvent(
            source="statsbomb",
            source_event_id=UUID(event_id),
            source_record_index=index - 1,
            source_event_index=index,
            match_id=100,
            team_id=team_id,
            player_id=player_id,
            possession_id=possession_id,
            possession_team_id=possession_team_id,
            event_type=event_type,
            event_subtype=None,
            period=1,
            timestamp=timestamp,
            minute=0,
            second=timestamp.second,
            duration=0.5,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            outcome=outcome,
            body_part="right_foot",
            recipient_id=None,
            xg=None,
            play_pattern="regular_play",
            under_pressure=False,
            counterpress=False,
            related_event_ids=(),
            raw_details=payload,
        )

    @staticmethod
    def _inputs(
        events: list[CanonicalEvent],
        *,
        frames: list[Canonical360Frame] | None = None,
        home: dict[int, int] | None = None,
        away: dict[int, int] | None = None,
    ) -> SpadlInput:
        frames = frames or []
        return SpadlInput(
            events=tuple(events),
            lineup_intervals=(),
            three_sixty_by_event={
                (frame.source, frame.source_event_id): frame for frame in frames
            },
            home_team_by_match=home or {100: 10},
            away_team_by_match=away or {100: 20},
            report=SpadlInputValidationReport(
                match_count=len({event.match_id for event in events}),
                event_count=len(events),
                lineup_interval_count=0,
                three_sixty_count=len(frames),
                events_with_subtype=0,
                issues=(),
            ),
        )


class FakeDatabaseResult:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows[0] if self.rows else None


class FakeAnalyticsConnection:
    def __init__(self) -> None:
        self.batch_rows: list[list[tuple[object, ...]]] = []

    def execute(
        self, sql: str, params: tuple[object, ...] = ()
    ) -> FakeDatabaseResult:
        del params
        if "information_schema.tables" in sql:
            return FakeDatabaseResult(
                [(table,) for table in PostgresAnalyticsWriter.REQUIRED_TABLES]
            )
        if "INSERT INTO analytics_runs" in sql:
            return FakeDatabaseResult([(7,)])
        return FakeDatabaseResult([])

    def cursor(self) -> FakeAnalyticsConnection:
        return self

    def executemany(
        self, sql: str, rows: list[tuple[object, ...]]
    ) -> None:
        del sql
        self.batch_rows.append(rows)


if __name__ == "__main__":
    unittest.main()
