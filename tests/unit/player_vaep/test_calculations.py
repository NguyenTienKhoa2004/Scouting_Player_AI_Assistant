from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.player_vaep.calculations import (  # noqa: E402
    MatchContext,
    PlayerVaepAggregator,
    calculate_player_minutes,
    match_duration_seconds,
)


class PlayerMinuteTests(unittest.TestCase):
    def test_substitution_intervals_determine_minutes_played(self) -> None:
        lineups = [
            {
                "team_id": 10,
                "lineup": [
                    {
                        "player_id": 1,
                        "positions": [
                            {
                                "position": "Forward",
                                "from": "00:00",
                                "to": "60:00",
                            }
                        ],
                    },
                    {
                        "player_id": 2,
                        "positions": [
                            {
                                "position": "Forward",
                                "from": "60:00",
                                "to": None,
                            }
                        ],
                    },
                ],
            }
        ]

        minutes = calculate_player_minutes(lineups, 95 * 60)

        self.assertEqual(minutes[(1, 10)].minutes_played, 60.0)
        self.assertEqual(minutes[(2, 10)].minutes_played, 35.0)

    def test_counts_added_time_and_merges_tactical_position_intervals(self) -> None:
        duration = 95 * 60
        lineups = [
            {
                "team_id": 10,
                "lineup": [
                    {
                        "player_id": 1,
                        "positions": [
                            {
                                "position": "Center Back",
                                "from": "00:00",
                                "to": "45:00",
                            },
                            {
                                "position": "Right Back",
                                "from": "45:00",
                                "to": None,
                            },
                        ],
                    },
                    {
                        "player_id": 2,
                        "positions": [
                            {
                                "position": "Center Forward",
                                "from": "60:00",
                                "to": None,
                            }
                        ],
                    },
                ],
            }
        ]

        minutes = calculate_player_minutes(lineups, duration)

        self.assertEqual(minutes[(1, 10)].minutes_played, 95.0)
        self.assertEqual(minutes[(1, 10)].primary_position, "Right Back")
        self.assertEqual(minutes[(2, 10)].minutes_played, 35.0)

    def test_match_duration_excludes_penalty_shootout(self) -> None:
        events = [
            {"period": 2, "minute": 94, "second": 30},
            {"period": 5, "minute": 121, "second": 10},
        ]

        self.assertEqual(match_duration_seconds(events), 94 * 60 + 30)

    def test_ignores_one_reversed_interval_but_keeps_valid_intervals(self) -> None:
        lineups = [
            {
                "team_id": 10,
                "lineup": [
                    {
                        "player_id": 1,
                        "positions": [
                            {"position": "Wing", "from": "107:44", "to": "70:46"},
                            {"position": "Midfield", "from": "70:46", "to": "83:01"},
                        ],
                    }
                ],
            }
        ]

        minutes = calculate_player_minutes(lineups, 120 * 60)

        self.assertAlmostEqual(minutes[(1, 10)].minutes_played, 12.25)
        self.assertEqual(minutes[(1, 10)].primary_position, "Midfield")


class PlayerVaepAggregatorTests(unittest.TestCase):
    def test_per_90_uses_actual_minutes_and_minimum_minutes_guard(self) -> None:
        frame = pd.DataFrame(
            {
                "player_id": [1],
                "team_id": [10],
                "type_name": ["pass"],
                "offensive_value": [0.4],
                "defensive_value": [0.1],
                "vaep_value": [0.5],
            }
        )
        minutes = calculate_player_minutes(
            [
                {
                    "team_id": 10,
                    "lineup": [
                        {
                            "player_id": 1,
                            "positions": [
                                {
                                    "position": "Midfield",
                                    "from": "00:00",
                                    "to": "45:00",
                                }
                            ],
                        }
                    ],
                }
            ],
            90 * 60,
        )
        aggregator = PlayerVaepAggregator(minimum_minutes=60)

        aggregator.add_match(frame, MatchContext(100, 1, 2), minutes)
        row = next(
            item
            for item in aggregator.rows()
            if item["aggregation_level"] == "competition_season"
            and item["action_type"] == "all"
        )

        self.assertEqual(row["minutes_played"], 45.0)
        self.assertAlmostEqual(row["vaep_per_90"], 1.0)
        self.assertFalse(row["minimum_minutes_eligible"])

    def test_aggregates_known_players_and_keeps_missing_minutes_auditable(self) -> None:
        frame = pd.DataFrame(
            {
                "player_id": [1, 1, 2, None],
                "team_id": [10, 10, 20, 20],
                "type_name": ["pass", "shot", "pass", "pass"],
                "offensive_value": [0.1, 0.2, -0.1, 0.5],
                "defensive_value": [0.0, 0.1, 0.0, 0.0],
                "vaep_value": [0.1, 0.3, -0.1, 0.5],
            }
        )
        minutes = calculate_player_minutes(
            [
                {
                    "team_id": 10,
                    "lineup": [
                        {
                            "player_id": 1,
                            "positions": [
                                {
                                    "position": "Midfield",
                                    "from": "00:00",
                                    "to": None,
                                }
                            ],
                        }
                    ],
                }
            ],
            90 * 60,
        )
        aggregator = PlayerVaepAggregator(minimum_minutes=60)

        aggregator.add_match(frame, MatchContext(100, 1, 2), minutes)
        rows = aggregator.rows()
        player_one = next(
            row
            for row in rows
            if row["aggregation_level"] == "competition_season"
            and row["player_id"] == 1
            and row["action_type"] == "all"
        )
        player_two = next(
            row
            for row in rows
            if row["aggregation_level"] == "competition_season"
            and row["player_id"] == 2
            and row["action_type"] == "all"
        )

        self.assertEqual(player_one["action_count"], 2)
        self.assertAlmostEqual(player_one["player_vaep"], 0.4)
        self.assertAlmostEqual(player_one["vaep_per_90"], 0.4)
        self.assertTrue(player_one["minimum_minutes_eligible"])
        self.assertEqual(player_two["position"], "Unknown")
        self.assertIsNone(player_two["vaep_per_90"])
        self.assertEqual(aggregator.unknown_player_action_count, 1)
        self.assertEqual(aggregator.known_player_actions_without_minutes, 1)


if __name__ == "__main__":
    unittest.main()
