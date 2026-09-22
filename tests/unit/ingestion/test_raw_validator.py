from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.ingestion import (  # noqa: E402
    RawStatsBombValidationError,
    RawStatsBombValidator,
    StatsBombRawReader,
)
from pitchpulse.ingestion.run import ingest_selection  # noqa: E402


class RawStatsBombValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)
        (self.data_root / "matches" / "43").mkdir(parents=True)
        (self.data_root / "events").mkdir()
        (self.data_root / "lineups").mkdir()
        (self.data_root / "three-sixty").mkdir()

        self.event_id = "13b63722-8099-4cdf-818a-d2df3036a633"
        self.event = {
            "id": self.event_id,
            "index": 1,
            "period": 1,
            "timestamp": "00:00:10.000",
            "minute": 0,
            "second": 10,
            "type": {"id": 30, "name": "Pass"},
            "possession": 1,
            "possession_team": {"id": 10, "name": "Home"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 10, "name": "Home"},
            "player": {"id": 100, "name": "Player One"},
            "location": [60.0, 40.0],
            "pass": {
                "recipient": {"id": 101, "name": "Player Two"},
                "end_location": [70.0, 42.0],
            },
        }
        self._write_fixture()
        self.reader = StatsBombRawReader(self.data_root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_accepts_valid_raw_selection(self) -> None:
        report = RawStatsBombValidator().validate(self.reader)

        self.assertTrue(report.is_valid)
        self.assertTrue(report.is_ingestible)
        self.assertEqual(report.match_count, 1)
        self.assertEqual(report.event_count, 1)
        self.assertEqual(report.issues, ())

    def test_rejects_raw_coordinate_and_player_team_mismatch(self) -> None:
        self.event["location"] = [121.0, 40.0]
        self.event["team"] = {"id": 20, "name": "Away"}
        self._write_events()

        report = RawStatsBombValidator().validate(self.reader)

        self.assertIn("number_out_of_range", self._codes(report))
        self.assertIn("player_team_mismatch", self._codes(report))
        self.assertFalse(report.is_ingestible)

    def test_rejects_missing_related_event(self) -> None:
        self.event["related_events"] = [
            "23b63722-8099-4cdf-818a-d2df3036a633"
        ]
        self._write_events()

        report = RawStatsBombValidator().validate(self.reader)

        self.assertIn("related_event_not_found", self._codes(report))
        self.assertFalse(report.is_ingestible)

    def test_record_level_issue_is_deferred_to_canonical_quarantine(self) -> None:
        del self.event["pass"]["end_location"]
        self._write_events()

        report = RawStatsBombValidator().validate(self.reader)

        self.assertIn("location_required", self._codes(report))
        self.assertFalse(report.is_valid)
        self.assertTrue(report.is_ingestible)
        self.assertEqual(report.deferred_issue_count, 1)
        self.assertEqual(report.blocking_issues, ())

    def test_ingestion_is_blocked_before_database_writes(self) -> None:
        self.event["related_events"] = [
            "23b63722-8099-4cdf-818a-d2df3036a633"
        ]
        self._write_events()

        connection = ReadOnlyCheckpointConnection()
        with self.assertRaises(RawStatsBombValidationError) as context:
            ingest_selection(
                connection,
                reader=self.reader,
                dataset_id="test-dataset",
                source_version="test-version",
                manifest_path="test-manifest.json",
            )

        self.assertIn(
            "related_event_not_found", self._codes(context.exception.report)
        )
        self.assertEqual(connection.statements, ["checkpoint_select"])

    def test_incremental_validation_only_reads_selected_matches(self) -> None:
        matches_path = self.data_root / "matches" / "43" / "106.json"
        matches = json.loads(matches_path.read_text(encoding="utf-8"))
        second_match = dict(matches[0])
        second_match["match_id"] = 1002
        self._write_json(matches_path, [matches[0], second_match])
        lineups = json.loads(
            (self.data_root / "lineups" / "1001.json").read_text(encoding="utf-8")
        )
        self._write_json(self.data_root / "lineups" / "1002.json", lineups)
        invalid_event = dict(self.event)
        invalid_event["id"] = "23b63722-8099-4cdf-818a-d2df3036a633"
        invalid_event["related_events"] = [
            "33b63722-8099-4cdf-818a-d2df3036a633"
        ]
        self._write_json(
            self.data_root / "events" / "1002.json",
            [invalid_event],
        )

        incremental = RawStatsBombValidator().validate(self.reader, (1001,))
        complete = RawStatsBombValidator().validate(self.reader)

        self.assertTrue(incremental.is_ingestible)
        self.assertEqual(incremental.match_count, 1)
        self.assertFalse(complete.is_ingestible)

    def test_dry_run_plans_without_database_writes(self) -> None:
        connection = ReadOnlyCheckpointConnection()

        result = ingest_selection(
            connection,
            reader=self.reader,
            dataset_id="test-dataset",
            source_version="test-version",
            manifest_path="test-manifest.json",
            dry_run=True,
        )

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["planned_match_count"], 1)
        self.assertIsNone(result["ingestion_run_id"])
        self.assertEqual(connection.statements, ["checkpoint_select"])

    def _write_fixture(self) -> None:
        self._write_json(
            self.data_root / "matches" / "43" / "106.json",
            [
                {
                    "match_id": 1001,
                    "competition": {
                        "competition_id": 43,
                        "competition_name": "Competition",
                    },
                    "season": {"season_id": 106, "season_name": "Season"},
                    "home_team": {
                        "home_team_id": 10,
                        "home_team_name": "Home",
                    },
                    "away_team": {
                        "away_team_id": 20,
                        "away_team_name": "Away",
                    },
                    "home_score": 1,
                    "away_score": 0,
                    "match_date": "2022-01-01",
                }
            ],
        )
        self._write_json(
            self.data_root / "lineups" / "1001.json",
            [
                {
                    "team_id": 10,
                    "team_name": "Home",
                    "lineup": [
                        {
                            "player_id": 100,
                            "player_name": "Player One",
                            "positions": [],
                        },
                        {
                            "player_id": 101,
                            "player_name": "Player Two",
                            "positions": [],
                        },
                    ],
                },
                {
                    "team_id": 20,
                    "team_name": "Away",
                    "lineup": [
                        {
                            "player_id": 200,
                            "player_name": "Player Three",
                            "positions": [],
                        }
                    ],
                },
            ],
        )
        self._write_events()

    def _write_events(self) -> None:
        self._write_json(self.data_root / "events" / "1001.json", [self.event])

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _codes(report: object) -> set[str]:
        return {issue.code for issue in report.issues}  # type: ignore[attr-defined]


class FakeResult:
    def fetchall(self) -> list[tuple[object, ...]]:
        return []


class ReadOnlyCheckpointConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(
        self, sql: str, params: tuple[object, ...] = ()
    ) -> FakeResult:
        del params
        if "FROM meta.ingestion_checkpoints" not in sql:
            raise AssertionError(f"unexpected database write: {sql}")
        self.statements.append("checkpoint_select")
        return FakeResult()


if __name__ == "__main__":
    unittest.main()
