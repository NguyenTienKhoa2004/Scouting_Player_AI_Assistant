from __future__ import annotations

import os
import sys
import unittest
from datetime import date, time
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from matchmind.ingestion.postgres_writer import PostgresDataWriter  # noqa: E402
from matchmind.ingestion.normalizer import CanonicalEvent  # noqa: E402
from matchmind.ingestion.lineup_normalizer import (  # noqa: E402
    CanonicalLineupInterval,
)
from matchmind.ingestion.reader import RawRecord  # noqa: E402
from matchmind.ingestion.three_sixty_normalizer import (  # noqa: E402
    Canonical360Frame,
)
from matchmind.spadl.input_reader import PostgresSpadlInputReader  # noqa: E402


TEST_DATABASE_URL = os.environ.get("MATCHMIND_INTEGRATION_DATABASE_URL")
MIGRATIONS = tuple(
    PROJECT_ROOT / "infra" / "db" / "migrations" / name
    for name in (
        "001_data_foundation.up.sql",
        "002_vaep_event_enrichment.up.sql",
        "003_spadl_analytics.up.sql",
        "004_medallion_silver.up.sql",
        "005_medallion_gold.up.sql",
    )
)


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "set MATCHMIND_INTEGRATION_DATABASE_URL to run PostgreSQL integration tests",
)
class SilverPostgresIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        assert TEST_DATABASE_URL is not None
        self.database_name = f"matchmind_silver_test_{uuid4().hex[:12]}"
        self.admin = psycopg.connect(TEST_DATABASE_URL, autocommit=True)
        self.admin.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database_name))
        )
        parameters = conninfo_to_dict(TEST_DATABASE_URL)
        parameters["dbname"] = self.database_name
        self.database_url = make_conninfo(**parameters)
        self.connection = psycopg.connect(self.database_url, autocommit=True)
        for migration in MIGRATIONS:
            self.connection.execute(migration.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.connection.close()
        self.admin.execute(
            sql.SQL("DROP DATABASE {}").format(sql.Identifier(self.database_name))
        )
        self.admin.close()

    def test_migration_and_writers_use_silver_boundaries(self) -> None:
        expected = (
            "meta.ingestion_runs",
            "meta.analytics_runs",
            "quarantine.invalid_events",
            "silver.teams",
            "silver.matches",
            "silver.players",
            "silver.events",
            "silver.player_match_intervals",
            "silver.event_360",
            "gold.analytics_actions",
            "gold.analytics_action_features",
        )
        row = self.connection.execute(
            "SELECT " + ", ".join(f"to_regclass('{name}')" for name in expected)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertTrue(all(value is not None for value in row or ()))
        self.assertIsNone(
            self.connection.execute("SELECT to_regclass('public.events')").fetchone()[0]
        )
        self.assertIsNone(
            self.connection.execute(
                "SELECT to_regclass('public.analytics_actions')"
            ).fetchone()[0]
        )

        writer = PostgresDataWriter(self.connection)
        writer.require_schema()
        run_id = writer.create_ingestion_run(
            dataset_id="integration-fixture-v1",
            source="statsbomb",
            source_version="a" * 40,
            competition_id=1,
            season_id=1,
            manifest_path="configs/datasets/test.json",
        )
        writer.upsert_teams(((10, "Home"), (20, "Away")))
        writer.upsert_players(((100, "Player", 10, "Center Forward"),))
        writer.upsert_match(
            match_id=1000,
            competition_id=1,
            competition="Integration League",
            season_id=1,
            season="2025/2026",
            home_team_id=10,
            away_team_id=20,
            home_score=0,
            away_score=0,
            match_date=date(2026, 1, 1),
        )
        event_id = uuid4()
        event = CanonicalEvent(
            source="statsbomb",
            source_event_id=event_id,
            source_record_index=0,
            source_event_index=1,
            match_id=1000,
            team_id=10,
            player_id=100,
            possession_id=1,
            possession_team_id=10,
            event_type="pass",
            event_subtype=None,
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
            recipient_id=None,
            xg=None,
            play_pattern="regular_play",
            under_pressure=False,
            counterpress=False,
            related_event_ids=(),
            raw_details={},
        )
        interval = CanonicalLineupInterval(
            source="statsbomb",
            source_record_index=0,
            match_id=1000,
            team_id=10,
            player_id=100,
            position_id=23,
            position="Center Forward",
            from_seconds=0,
            to_seconds=5400,
            from_period=1,
            to_period=2,
            start_reason="Starting XI",
            end_reason="Final Whistle",
            raw_details={},
        )
        frame = Canonical360Frame(
            source="statsbomb",
            source_event_id=event_id,
            source_record_index=0,
            match_id=1000,
            visible_area=(),
            freeze_frame=(),
            raw_details={},
        )
        self.assertEqual(writer.upsert_events(run_id, (event,)).inserted, 1)
        self.assertEqual(writer.upsert_events(run_id, (event,)).updated, 1)
        self.assertEqual(
            writer.upsert_player_match_intervals(run_id, (interval,)).inserted,
            1,
        )
        self.assertEqual(
            writer.upsert_player_match_intervals(run_id, (interval,)).updated,
            1,
        )
        self.assertEqual(writer.upsert_three_sixty(run_id, (frame,)).inserted, 1)
        self.assertEqual(writer.upsert_three_sixty(run_id, (frame,)).updated, 1)
        writer.insert_invalid_event(
            ingestion_run_id=run_id,
            record=RawRecord(
                kind="event",
                match_id=1000,
                source_file=PROJECT_ROOT / "fixture.json",
                source_record_index=0,
                payload={"id": "invalid"},
            ),
            reason_code="fixture_invalid",
            reason="integration fixture",
            project_root=PROJECT_ROOT,
        )

        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM silver.matches").fetchone()[0],
            1,
        )
        self.assertEqual(writer.count_events_for_matches((1000,)), 1)
        self.assertEqual(writer.count_lineup_intervals_for_matches((1000,)), 1)
        self.assertEqual(writer.count_three_sixty_for_matches((1000,)), 1)
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM quarantine.invalid_events"
            ).fetchone()[0],
            1,
        )
        reader = PostgresSpadlInputReader(self.connection)
        self.assertEqual(reader.validate_schema(), ())
        loaded = reader.load((1000,))
        self.assertEqual(len(loaded.events), 1)
        self.assertEqual(len(loaded.lineup_intervals), 1)
        self.assertEqual(len(loaded.three_sixty_by_event), 1)

    def test_down_migration_restores_the_pre_silver_layout(self) -> None:
        gold_down = (
            PROJECT_ROOT
            / "infra"
            / "db"
            / "migrations"
            / "005_medallion_gold.down.sql"
        )
        self.connection.execute(gold_down.read_text(encoding="utf-8"))

        down = (
            PROJECT_ROOT
            / "infra"
            / "db"
            / "migrations"
            / "004_medallion_silver.down.sql"
        )
        self.connection.execute(down.read_text(encoding="utf-8"))

        self.assertIsNotNone(
            self.connection.execute("SELECT to_regclass('public.events')").fetchone()[0]
        )
        self.assertIsNone(
            self.connection.execute("SELECT to_regclass('silver.events')").fetchone()[0]
        )
        self.assertIsNotNone(
            self.connection.execute(
                "SELECT to_regclass('public.analytics_actions')"
            ).fetchone()[0]
        )
        self.assertIsNone(
            self.connection.execute(
                "SELECT to_regclass('gold.analytics_actions')"
            ).fetchone()[0]
        )


if __name__ == "__main__":
    unittest.main()
