from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date, time
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.ingestion.postgres_writer import PostgresDataWriter  # noqa: E402
from pitchpulse.ingestion import (  # noqa: E402
    IngestionCounts,
    IngestionPlan,
    PlannedMatch,
    StatsBombIngestionService,
    StatsBombRawReader,
)
from pitchpulse.ingestion.run import ingest_planned_matches, ingest_selection  # noqa: E402
from pitchpulse.ingestion.normalizer import CanonicalEvent  # noqa: E402
from pitchpulse.ingestion.lineup_normalizer import (  # noqa: E402
    CanonicalLineupInterval,
)
from pitchpulse.ingestion.reader import RawRecord  # noqa: E402
from pitchpulse.ingestion.three_sixty_normalizer import (  # noqa: E402
    Canonical360Frame,
)
from pitchpulse.spadl.input_reader import PostgresSpadlInputReader  # noqa: E402


TEST_DATABASE_URL = os.environ.get("PITCHPULSE_INTEGRATION_DATABASE_URL")
MIGRATIONS = tuple(
    PROJECT_ROOT / "infra" / "db" / "migrations" / name
    for name in (
        "001_data_foundation.up.sql",
        "002_vaep_event_enrichment.up.sql",
        "003_spadl_analytics.up.sql",
        "004_medallion_silver.up.sql",
        "005_medallion_gold.up.sql",
        "006_vaep_modeling.up.sql",
        "007_spadl_table_names.up.sql",
        "008_ingestion_checkpoints.up.sql",
        "009_silver_snapshot_reconciliation.up.sql",
        "010_ingestion_match_metrics.up.sql",
    )
)


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "set PITCHPULSE_INTEGRATION_DATABASE_URL to run PostgreSQL integration tests",
)
class SilverPostgresIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        assert TEST_DATABASE_URL is not None
        self.database_name = f"pitchpulse_silver_test_{uuid4().hex[:12]}"
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
            "meta.ingestion_checkpoints",
            "quarantine.invalid_events",
            "silver.teams",
            "silver.matches",
            "silver.players",
            "silver.events",
            "silver.player_match_intervals",
            "silver.event_360",
            "gold.spadl_actions",
            "gold.spadl_action_features",
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
        checkpoint_hash = "a" * 64
        writer.upsert_ingestion_checkpoint(
            source="statsbomb",
            dataset_id="integration-fixture-v1",
            competition_id=1,
            season_id=1,
            match_id=1000,
            content_hash=checkpoint_hash,
            source_version="a" * 40,
            ingestion_run_id=run_id,
        )
        checkpoint = self.connection.execute(
            """
            SELECT content_hash, source_version, processed_at
            FROM meta.ingestion_checkpoints
            WHERE source = %s AND dataset_id = %s AND match_id = %s
            """,
            ("statsbomb", "integration-fixture-v1", 1000),
        ).fetchone()
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint[0], checkpoint_hash)
        self.assertEqual(checkpoint[1], "a" * 40)
        self.assertIsNotNone(checkpoint[2])
        self.assertEqual(
            writer.load_ingestion_checkpoint_hashes(
                source="statsbomb",
                competition_id=1,
                season_id=1,
            ),
            {1000: checkpoint_hash},
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

        removed_event = replace(
            event,
            source_event_id=uuid4(),
            source_event_index=2,
        )
        writer.upsert_events(run_id, (removed_event,))
        reconciliation_run_id = writer.create_ingestion_run(
            dataset_id="integration-fixture-v1",
            source="statsbomb",
            source_version="b" * 40,
            competition_id=1,
            season_id=1,
            manifest_path="configs/datasets/test.json",
        )
        with self.connection.transaction():
            writer.deactivate_match_snapshot(
                ingestion_run_id=reconciliation_run_id,
                source="statsbomb",
                match_id=1000,
            )
            writer.upsert_events(reconciliation_run_id, (event,))
            writer.upsert_player_match_intervals(
                reconciliation_run_id,
                (interval,),
            )
            writer.upsert_ingestion_checkpoint(
                source="statsbomb",
                dataset_id="integration-fixture-v1",
                competition_id=1,
                season_id=1,
                match_id=1000,
                content_hash="b" * 64,
                source_version="b" * 40,
                ingestion_run_id=reconciliation_run_id,
            )

        event_states = self.connection.execute(
            """
            SELECT source_event_id, is_active, removed_ingestion_run_id
            FROM silver.events
            WHERE match_id = 1000
            ORDER BY source_event_index
            """
        ).fetchall()
        self.assertEqual(event_states[0], (event.source_event_id, True, None))
        self.assertEqual(
            event_states[1],
            (removed_event.source_event_id, False, reconciliation_run_id),
        )
        self.assertEqual(writer.count_events_for_matches((1000,)), 1)
        self.assertEqual(writer.count_lineup_intervals_for_matches((1000,)), 1)
        self.assertEqual(writer.count_three_sixty_for_matches((1000,)), 0)
        reconciled_input = reader.load((1000,))
        self.assertEqual(len(reconciled_input.events), 1)
        self.assertEqual(len(reconciled_input.lineup_intervals), 1)
        self.assertEqual(len(reconciled_input.three_sixty_by_event), 0)

    def test_match_transactions_commit_checkpoint_atomically_and_resume(self) -> None:
        writer = PostgresDataWriter(self.connection)
        run_id = writer.create_ingestion_run(
            dataset_id="transaction-fixture-v1",
            source="statsbomb",
            source_version="a" * 40,
            competition_id=1,
            season_id=1,
            manifest_path="configs/datasets/test.json",
        )
        first = PlannedMatch(2001, "new", "1" * 64, None)
        second = PlannedMatch(2002, "new", "2" * 64, None)
        plan = IngestionPlan((first, second))

        class FixtureService:
            def __init__(self, fail_match_id: int | None) -> None:
                self.fail_match_id = fail_match_id

            def ingest_match(
                service_self, ingestion_run_id: int, match_id: int
            ) -> IngestionCounts:
                del ingestion_run_id
                home_team_id = match_id * 10
                away_team_id = home_team_id + 1
                writer.upsert_teams(
                    (
                        (home_team_id, f"Home {match_id}"),
                        (away_team_id, f"Away {match_id}"),
                    )
                )
                writer.upsert_match(
                    match_id=match_id,
                    competition_id=1,
                    competition="Transaction League",
                    season_id=1,
                    season="2025/2026",
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_score=0,
                    away_score=0,
                    match_date=date(2026, 1, 1),
                )
                if match_id == service_self.fail_match_id:
                    raise RuntimeError("fixture match failure")
                return IngestionCounts()

        with self.assertRaisesRegex(RuntimeError, "fixture match failure"):
            ingest_planned_matches(
                self.connection,
                writer=writer,
                service=FixtureService(2002),
                plan=plan,
                run_id=run_id,
                source="statsbomb",
                dataset_id="transaction-fixture-v1",
                source_version="a" * 40,
                competition_id=1,
                season_id=1,
            )

        persisted_matches = self.connection.execute(
            "SELECT id FROM silver.matches ORDER BY id"
        ).fetchall()
        persisted_checkpoints = self.connection.execute(
            "SELECT match_id FROM meta.ingestion_checkpoints ORDER BY match_id"
        ).fetchall()
        self.assertEqual(persisted_matches, [(2001,)])
        self.assertEqual(persisted_checkpoints, [(2001,)])

        resume_plan = IngestionPlan((second,))
        ingest_planned_matches(
            self.connection,
            writer=writer,
            service=FixtureService(None),
            plan=resume_plan,
            run_id=run_id,
            source="statsbomb",
            dataset_id="transaction-fixture-v1",
            source_version="a" * 40,
            competition_id=1,
            season_id=1,
        )

        self.assertEqual(
            self.connection.execute(
                "SELECT match_id FROM meta.ingestion_checkpoints ORDER BY match_id"
            ).fetchall(),
            [(2001,), (2002,)],
        )

    def test_incremental_ingestion_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            (data_root / "matches" / "1").mkdir(parents=True)
            (data_root / "events").mkdir()
            (data_root / "lineups").mkdir()
            (data_root / "three-sixty").mkdir()
            match_rows: list[dict[str, object]] = []
            events_by_match: dict[int, list[dict[str, object]]] = {}

            def event(match_id: int, index: int) -> dict[str, object]:
                return {
                    "id": str(UUID(int=match_id * 10 + index)),
                    "index": index,
                    "period": 1,
                    "timestamp": f"00:00:{index:02d}.000",
                    "minute": 0,
                    "second": index,
                    "duration": 0.5,
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

            lineups = [
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
            ]

            def add_match(match_id: int, *, two_events: bool = False) -> None:
                match_rows.append(
                    {
                        "match_id": match_id,
                        "competition": {
                            "competition_id": 1,
                            "competition_name": "Integration League",
                        },
                        "season": {"season_id": 1, "season_name": "2025/2026"},
                        "home_team": {
                            "home_team_id": 10,
                            "home_team_name": "Home",
                        },
                        "away_team": {
                            "away_team_id": 20,
                            "away_team_name": "Away",
                        },
                        "home_score": 0,
                        "away_score": 0,
                        "match_date": "2026-01-01",
                    }
                )
                events_by_match[match_id] = [event(match_id, 1)]
                if two_events:
                    events_by_match[match_id].append(event(match_id, 2))
                write_json(data_root / "events" / f"{match_id}.json", events_by_match[match_id])
                write_json(data_root / "lineups" / f"{match_id}.json", lineups)

            def write_json(path: Path, value: object) -> None:
                path.write_text(json.dumps(value), encoding="utf-8")

            def write_matches() -> None:
                write_json(data_root / "matches" / "1" / "1.json", match_rows)

            def run_ingestion(
                version: str,
                *,
                dataset_id: str,
                force: bool = False,
            ) -> dict[str, object]:
                return ingest_selection(
                    self.connection,
                    reader=StatsBombRawReader(
                        data_root=data_root,
                        competition_id=1,
                        season_id=1,
                    ),
                    dataset_id=dataset_id,
                    source_version=version,
                    manifest_path="configs/datasets/integration.json",
                    force=force,
                )

            add_match(1, two_events=True)
            write_matches()
            initial = run_ingestion(
                "1" * 40,
                dataset_id="incremental-fixture-v1",
            )
            self.assertEqual(initial["new_match_count"], 1)
            self.assertEqual(initial["ingested_match_count"], 1)

            for match_id in range(2, 52):
                add_match(match_id)
            write_matches()
            added = run_ingestion(
                "2" * 40,
                dataset_id="incremental-fixture-v2",
            )
            self.assertEqual(added["new_match_count"], 50)
            self.assertEqual(added["unchanged_match_count"], 1)
            self.assertEqual(added["ingested_match_count"], 50)
            added_metrics = self.connection.execute(
                """
                SELECT discovered_matches, new_matches, changed_matches,
                       skipped_matches, removed_matches
                FROM meta.ingestion_runs WHERE id = %s
                """,
                (added["ingestion_run_id"],),
            ).fetchone()
            self.assertEqual(added_metrics, (51, 50, 0, 1, 0))

            unchanged = run_ingestion(
                "2" * 40,
                dataset_id="incremental-fixture-v2",
            )
            self.assertEqual(unchanged["ingested_match_count"], 0)
            self.assertEqual(unchanged["unchanged_match_count"], 51)

            events_by_match[1][0]["duration"] = 1.0
            write_json(data_root / "events" / "1.json", events_by_match[1])
            changed = run_ingestion(
                "3" * 40,
                dataset_id="incremental-fixture-v3",
            )
            self.assertEqual(changed["changed_match_count"], 1)
            self.assertEqual(changed["ingested_match_count"], 1)

            add_match(52)
            add_match(53)
            write_matches()
            original_ingest_match = StatsBombIngestionService.ingest_match

            def fail_on_53(
                service: StatsBombIngestionService,
                ingestion_run_id: int,
                match_id: int,
            ) -> IngestionCounts:
                if match_id == 53:
                    raise RuntimeError("intentional integration failure")
                return original_ingest_match(service, ingestion_run_id, match_id)

            with patch.object(
                StatsBombIngestionService,
                "ingest_match",
                new=fail_on_53,
            ):
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    run_ingestion(
                        "4" * 40,
                        dataset_id="incremental-fixture-v4",
                    )

            checkpoint_ids = {
                int(row[0])
                for row in self.connection.execute(
                    """
                    SELECT match_id FROM meta.ingestion_checkpoints
                    WHERE dataset_id = 'incremental-fixture-v4'
                    """
                ).fetchall()
            }
            self.assertIn(52, checkpoint_ids)
            self.assertNotIn(53, checkpoint_ids)
            resumed = run_ingestion(
                "4" * 40,
                dataset_id="incremental-fixture-v4",
            )
            self.assertEqual(resumed["new_match_count"], 1)
            self.assertEqual(resumed["ingested_match_count"], 1)

            events_by_match[1] = events_by_match[1][:1]
            write_json(data_root / "events" / "1.json", events_by_match[1])
            reconciled = run_ingestion(
                "5" * 40,
                dataset_id="incremental-fixture-v5",
            )
            self.assertEqual(reconciled["changed_match_count"], 1)
            event_counts = self.connection.execute(
                """
                SELECT COUNT(*), COUNT(*) FILTER (WHERE is_active)
                FROM silver.events WHERE match_id = 1
                """
            ).fetchone()
            self.assertEqual(event_counts, (2, 1))

            forced = run_ingestion(
                "5" * 40,
                dataset_id="incremental-fixture-v5",
                force=True,
            )
            self.assertEqual(forced["ingested_match_count"], 53)
            self.assertEqual(forced["new_match_count"], 0)
            self.assertEqual(forced["changed_match_count"], 0)
            active_events = self.connection.execute(
                "SELECT COUNT(*) FROM silver.events WHERE is_active"
            ).fetchone()[0]
            distinct_active_events = self.connection.execute(
                """
                SELECT COUNT(DISTINCT (source, source_event_id))
                FROM silver.events WHERE is_active
                """
            ).fetchone()[0]
            self.assertEqual(active_events, 53)
            self.assertEqual(distinct_active_events, 53)

            force_metrics = self.connection.execute(
                """
                SELECT discovered_matches, new_matches, changed_matches,
                       skipped_matches, removed_matches
                FROM meta.ingestion_runs WHERE id = %s
                """,
                (forced["ingestion_run_id"],),
            ).fetchone()
            self.assertEqual(force_metrics, (53, 0, 0, 0, 0))

            match_rows[:] = [
                row for row in match_rows if int(row["match_id"]) != 53
            ]
            write_matches()
            removed = run_ingestion(
                "6" * 40,
                dataset_id="incremental-fixture-v6",
            )
            self.assertEqual(removed["removed_match_count"], 1)
            self.assertEqual(
                self.connection.execute(
                    "SELECT COUNT(*) FROM meta.ingestion_checkpoints "
                    "WHERE source = 'statsbomb' AND match_id = 53"
                ).fetchone()[0],
                0,
            )

            repeated = run_ingestion(
                "6" * 40,
                dataset_id="incremental-fixture-v6",
            )
            self.assertEqual(repeated["removed_match_count"], 0)

    def test_down_migration_restores_the_pre_silver_layout(self) -> None:
        metrics_down = (
            PROJECT_ROOT
            / "infra"
            / "db"
            / "migrations"
            / "010_ingestion_match_metrics.down.sql"
        )
        self.connection.execute(metrics_down.read_text(encoding="utf-8"))

        reconciliation_down = (
            PROJECT_ROOT
            / "infra"
            / "db"
            / "migrations"
            / "009_silver_snapshot_reconciliation.down.sql"
        )
        self.connection.execute(reconciliation_down.read_text(encoding="utf-8"))

        checkpoints_down = (
            PROJECT_ROOT
            / "infra"
            / "db"
            / "migrations"
            / "008_ingestion_checkpoints.down.sql"
        )
        self.connection.execute(checkpoints_down.read_text(encoding="utf-8"))

        spadl_names_down = (
            PROJECT_ROOT
            / "infra"
            / "db"
            / "migrations"
            / "007_spadl_table_names.down.sql"
        )
        self.connection.execute(spadl_names_down.read_text(encoding="utf-8"))

        vaep_down = (
            PROJECT_ROOT
            / "infra"
            / "db"
            / "migrations"
            / "006_vaep_modeling.down.sql"
        )
        self.connection.execute(vaep_down.read_text(encoding="utf-8"))

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
                "SELECT to_regclass('gold.spadl_actions')"
            ).fetchone()[0]
        )


if __name__ == "__main__":
    unittest.main()
