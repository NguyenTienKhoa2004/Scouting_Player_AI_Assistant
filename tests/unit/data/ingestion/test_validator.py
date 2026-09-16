from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from datetime import time
from pathlib import Path
from types import MappingProxyType
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.data.ingestion import (  # noqa: E402
    Canonical360Frame,
    Canonical360Validator,
    CanonicalEvent,
    CanonicalEventValidator,
    EventValidationContext,
)


class CanonicalEventValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        context = EventValidationContext(
            match_team_ids=MappingProxyType({3857276: frozenset({788, 1833})}),
            match_player_ids=MappingProxyType({3857276: frozenset({5237, 6301})}),
        )
        self.validator = CanonicalEventValidator(context)
        self.shot = CanonicalEvent(
            source="statsbomb",
            source_event_id=UUID("13b63722-8099-4cdf-818a-d2df3036a633"),
            source_record_index=100,
            source_event_index=101,
            match_id=3857276,
            team_id=788,
            player_id=5237,
            possession_id=8,
            possession_team_id=788,
            event_type="shot",
            event_subtype="open_play",
            period=1,
            timestamp=time.fromisoformat("00:03:29.220"),
            minute=3,
            second=29,
            duration=0.4,
            x=69.9166666667,
            y=56.25,
            end_x=100.0,
            end_y=54.125,
            outcome="goal",
            body_part="right_foot",
            recipient_id=None,
            xg=0.024477394,
            play_pattern="regular_play",
            under_pressure=False,
            counterpress=False,
            related_event_ids=(),
            raw_details={},
        )

    def test_accepts_valid_shot(self) -> None:
        result = self.validator.validate(self.shot)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.issues, ())

    def test_rejects_missing_player_for_shot(self) -> None:
        result = self.validator.validate(replace(self.shot, player_id=None))
        self.assertIn("player_required", self._codes(result))

    def test_allows_missing_player_and_location_for_starting_xi(self) -> None:
        event = replace(
            self.shot,
            player_id=None,
            event_type="starting_xi",
            timestamp=time(0, 0),
            minute=0,
            second=0,
            x=None,
            y=None,
            end_x=None,
            end_y=None,
            outcome=None,
            xg=None,
        )
        result = self.validator.validate(event)
        self.assertTrue(result.is_valid)

    def test_rejects_out_of_range_coordinate(self) -> None:
        result = self.validator.validate(replace(self.shot, x=101.0))
        self.assertIn("start_x_out_of_range", self._codes(result))

    def test_rejects_inconsistent_period_time(self) -> None:
        result = self.validator.validate(
            replace(
                self.shot,
                period=2,
                timestamp=time(0, 3, 29, 220000),
                minute=3,
            )
        )
        self.assertIn("time_fields_inconsistent", self._codes(result))

    def test_rejects_player_not_in_match_lineup(self) -> None:
        result = self.validator.validate(replace(self.shot, player_id=999999))
        self.assertIn("player_not_in_match_lineup", self._codes(result))

    def test_rejects_xg_on_non_shot(self) -> None:
        event = replace(
            self.shot,
            event_type="pressure",
            end_x=None,
            end_y=None,
            outcome=None,
        )
        result = self.validator.validate(event)
        self.assertIn("xg_not_allowed", self._codes(result))

    def test_rejects_possession_team_not_in_match(self) -> None:
        result = self.validator.validate(
            replace(self.shot, possession_team_id=999999)
        )
        self.assertIn("possession_team_not_in_match", self._codes(result))

    def test_validates_three_sixty_link_and_geometry(self) -> None:
        frame = Canonical360Frame(
            source="statsbomb",
            source_event_id=self.shot.source_event_id,
            source_record_index=0,
            match_id=3857276,
            visible_area=(0.0, 0.0, 120.0, 0.0, 120.0, 80.0),
            freeze_frame=(
                {
                    "teammate": True,
                    "actor": True,
                    "keeper": False,
                    "location": [61.0, 40.0],
                },
            ),
            raw_details={},
        )

        result = Canonical360Validator().validate(
            frame, known_event_ids={self.shot.source_event_id}
        )

        self.assertTrue(result.is_valid)

    @staticmethod
    def _codes(result: object) -> set[str]:
        return {issue.code for issue in result.issues}  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
