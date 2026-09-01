from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matchmind.data import (  # noqa: E402
    NormalizationError,
    RawRecord,
    StatsBomb360Normalizer,
    StatsBombEventNormalizer,
    StatsBombLineupNormalizer,
)


class StatsBombEventNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = StatsBombEventNormalizer()

    def test_normalizes_shot_coordinates_outcome_and_xg(self) -> None:
        record = self._record(
            {
                "id": "13b63722-8099-4cdf-818a-d2df3036a633",
                "team": {"id": 788},
                "player": {"id": 5237},
                "type": {"name": "Shot"},
                "period": 1,
                "timestamp": "00:03:29.220",
                "minute": 3,
                "second": 29,
                "location": [83.9, 45.0],
                "shot": {
                    "end_location": [120.0, 43.3, 1.4],
                    "outcome": {"name": "Goal"},
                    "statsbomb_xg": 0.024477394,
                },
            }
        )

        event = self.normalizer.normalize(record)

        self.assertEqual(event.event_type, "shot")
        self.assertAlmostEqual(event.x, 69.9166666667)
        self.assertAlmostEqual(event.y, 56.25)
        self.assertEqual(event.end_x, 100.0)
        self.assertAlmostEqual(event.end_y, 54.125)
        self.assertEqual(event.outcome, "goal")
        self.assertEqual(event.xg, 0.024477394)

    def test_missing_pass_outcome_means_complete(self) -> None:
        record = self._record(
            {
                "id": "05a57281-9295-408a-9c4f-e32d7fad4d96",
                "team": {"id": 788},
                "player": {"id": 6301},
                "type": {"name": "Pass"},
                "period": 1,
                "timestamp": "00:00:00.460",
                "minute": 0,
                "second": 0,
                "location": [61.0, 40.1],
                "pass": {"end_location": [48.5, 41.3]},
            }
        )

        event = self.normalizer.normalize(record)

        self.assertEqual(event.event_type, "pass")
        self.assertEqual(event.outcome, "complete")
        self.assertIsNone(event.xg)

    def test_preserves_vaep_event_context_and_subtype(self) -> None:
        record = self._record(
            {
                "id": "05a57281-9295-408a-9c4f-e32d7fad4d96",
                "index": 5,
                "team": {"id": 788},
                "player": {"id": 6301},
                "possession": 2,
                "possession_team": {"id": 788},
                "type": {"name": "Pass"},
                "period": 1,
                "timestamp": "00:00:00.460",
                "minute": 0,
                "second": 0,
                "duration": 0.83469,
                "location": [61.0, 40.1],
                "play_pattern": {"name": "From Kick Off"},
                "under_pressure": True,
                "related_events": ["1267faf9-e033-4985-925a-2f9521d63bc0"],
                "pass": {
                    "end_location": [48.5, 41.3],
                    "type": {"name": "Kick Off"},
                    "body_part": {"name": "Left Foot"},
                    "recipient": {"id": 5234},
                },
            }
        )

        event = self.normalizer.normalize(record)

        self.assertEqual(event.source_event_index, 5)
        self.assertEqual(event.possession_id, 2)
        self.assertEqual(event.possession_team_id, 788)
        self.assertEqual(event.event_subtype, "kick_off")
        self.assertEqual(event.body_part, "left_foot")
        self.assertEqual(event.recipient_id, 5234)
        self.assertEqual(event.play_pattern, "from_kick_off")
        self.assertTrue(event.under_pressure)
        self.assertEqual(len(event.related_event_ids), 1)
        self.assertEqual(event.raw_details["pass"]["type"]["name"], "Kick Off")

    def test_normalizes_lineup_position_intervals(self) -> None:
        record = RawRecord(
            kind="lineup",
            match_id=3857276,
            source_file=Path("lineups/3857276.json"),
            source_record_index=0,
            payload={
                "team_id": 788,
                "lineup": [
                    {
                        "player_id": 3625,
                        "positions": [
                            {
                                "position_id": 21,
                                "position": "Left Wing",
                                "from": "00:00",
                                "to": "64:10",
                                "from_period": 1,
                                "to_period": 2,
                                "start_reason": "Starting XI",
                                "end_reason": "Substitution - Off (Tactical)",
                            }
                        ],
                    }
                ],
            },
        )

        intervals = StatsBombLineupNormalizer().normalize(record)

        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].from_seconds, 0)
        self.assertEqual(intervals[0].to_seconds, 3850)
        self.assertEqual(intervals[0].position, "Left Wing")

    def test_normalizes_three_sixty_snapshot(self) -> None:
        record = RawRecord(
            kind="three_sixty",
            match_id=3857276,
            source_file=Path("three-sixty/3857276.json"),
            source_record_index=0,
            payload={
                "event_uuid": "05a57281-9295-408a-9c4f-e32d7fad4d96",
                "visible_area": [0.0, 0.0, 120.0, 0.0, 120.0, 80.0],
                "freeze_frame": [
                    {
                        "teammate": True,
                        "actor": True,
                        "keeper": False,
                        "location": [61.0, 40.1],
                    }
                ],
            },
        )

        frame = StatsBomb360Normalizer().normalize(record)

        self.assertEqual(frame.match_id, 3857276)
        self.assertEqual(len(frame.visible_area), 6)
        self.assertTrue(frame.freeze_frame[0]["actor"])

    def test_allows_null_player_and_location_before_validation(self) -> None:
        record = self._record(
            {
                "id": "1f60fe1e-0340-49e2-975e-f16f54c5da1a",
                "team": {"id": 1833},
                "type": {"name": "Starting XI"},
                "period": 1,
                "timestamp": "00:00:00.000",
                "minute": 0,
                "second": 0,
            }
        )

        event = self.normalizer.normalize(record)

        self.assertEqual(event.event_type, "starting_xi")
        self.assertIsNone(event.player_id)
        self.assertIsNone(event.x)
        self.assertIsNone(event.y)

    def test_does_not_mutate_raw_payload(self) -> None:
        record = self._record(
            {
                "id": "05a57281-9295-408a-9c4f-e32d7fad4d96",
                "team": {"id": 788},
                "player": {"id": 6301},
                "type": {"name": "Pass"},
                "period": 1,
                "timestamp": "00:00:00.460",
                "minute": 0,
                "second": 0,
                "location": [61.0, 40.1],
                "pass": {"end_location": [48.5, 41.3]},
            }
        )
        original = copy.deepcopy(record.payload)

        self.normalizer.normalize(record)

        self.assertEqual(record.payload, original)

    def test_rejects_unknown_event_type(self) -> None:
        record = self._record(
            {
                "id": "05a57281-9295-408a-9c4f-e32d7fad4d96",
                "team": {"id": 788},
                "type": {"name": "New Source Type"},
                "period": 1,
                "timestamp": "00:00:00.000",
                "minute": 0,
                "second": 0,
            }
        )

        with self.assertRaisesRegex(NormalizationError, "unknown_event_type"):
            self.normalizer.normalize(record)

    @staticmethod
    def _record(payload: dict[str, object]) -> RawRecord:
        enriched = {
            "index": 1,
            "possession": 1,
            "possession_team": {"id": payload.get("team", {"id": 788})["id"]},
            "play_pattern": {"name": "Regular Play"},
        }
        enriched.update(payload)
        return RawRecord(
            kind="event",
            match_id=3857276,
            source_file=Path("events/3857276.json"),
            source_record_index=0,
            payload=enriched,
        )


if __name__ == "__main__":
    unittest.main()
