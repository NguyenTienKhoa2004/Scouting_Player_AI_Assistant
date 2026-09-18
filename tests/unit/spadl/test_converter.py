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
    ACTION_MAPPING_VERSION,
    ConversionError,
    EventToActionConverter,
    SpadlInput,
    SpadlInputValidationReport,
)
from matchmind.ingestion.normalizer import CanonicalEvent  # noqa: E402


class SocceractionConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = self._event(
            event_id="13b63722-8099-4cdf-818a-d2df3036a633",
            index=1,
            event_type="pass",
            source_type="Pass",
            timestamp="00:00:01.250",
            location=[60.0, 20.0],
            extra={
                "pass": {
                    "end_location": [72.0, 40.0],
                    "body_part": {"id": 40, "name": "Right Foot"},
                }
            },
        )

    def test_delegates_mapping_and_coordinates_to_socceraction(self) -> None:
        result = EventToActionConverter().convert(self._inputs([self.event]))
        action = result.actions[0]

        self.assertTrue(ACTION_MAPPING_VERSION.startswith("socceraction-"))
        self.assertEqual((action.match_id, action.action_id), (100, 0))
        self.assertEqual(action.source_event_id, self.event.source_event_id)
        self.assertEqual(action.time_seconds, 1.25)
        self.assertEqual(
            (action.type_name, action.result_name, action.bodypart_name),
            ("pass", "success", "foot_right"),
        )
        self.assertAlmostEqual(action.start_x, (60.0 - 0.05) / 120.0 * 105.0)
        self.assertAlmostEqual(
            action.start_y, 68.0 - (20.0 - 0.05) / 80.0 * 68.0
        )

    def test_socceraction_excludes_non_actions_and_splits_interception_pass(self) -> None:
        pressure = self._event(
            event_id="23b63722-8099-4cdf-818a-d2df3036a633",
            index=2,
            event_type="pressure",
            source_type="Pressure",
            timestamp="00:00:02.000",
            location=[72.0, 40.0],
        )
        interception_pass = self._event(
            event_id="33b63722-8099-4cdf-818a-d2df3036a633",
            index=3,
            event_type="pass",
            source_type="Pass",
            timestamp="00:00:03.000",
            location=[72.0, 40.0],
            extra={
                "pass": {
                    "type": {"id": 64, "name": "Interception"},
                    "end_location": [74.0, 40.0],
                }
            },
        )

        result = EventToActionConverter().convert(
            self._inputs([interception_pass, pressure, self.event])
        )

        self.assertEqual(
            [action.type_name for action in result.actions],
            ["pass", "interception", "pass"],
        )
        self.assertEqual([action.action_id for action in result.actions], [0, 1, 2])
        self.assertEqual(
            [action.source_action_index for action in result.actions], [0, 0, 1]
        )
        self.assertFalse(result.actions[1].synthetic)
        self.assertEqual(result.report.excluded_event_count, 1)
        self.assertEqual(
            dict(result.report.exclusions_by_reason),
            {"socceraction_non_action:pressure": 1},
        )

    def test_action_ids_reset_per_match_and_conversion_is_reproducible(self) -> None:
        other = replace(
            self.event,
            source_event_id=UUID("43b63722-8099-4cdf-818a-d2df3036a633"),
            match_id=200,
            team_id=30,
            possession_team_id=30,
            raw_details={
                **self.event.raw_details,
                "id": "43b63722-8099-4cdf-818a-d2df3036a633",
                "team": {"id": 30, "name": "Other"},
                "possession_team": {"id": 30, "name": "Other"},
            },
        )
        inputs = self._inputs(
            [other, self.event],
            home={100: 10, 200: 30},
            away={100: 20, 200: 40},
        )
        converter = EventToActionConverter()

        first = converter.convert(inputs)
        second = converter.convert(inputs)
        self.assertEqual(first, second)
        self.assertEqual(
            [(action.match_id, action.action_id) for action in first.actions],
            [(100, 0), (200, 0)],
        )

    def test_fails_when_socceraction_action_has_no_player(self) -> None:
        missing = replace(
            self.event,
            player_id=None,
            raw_details={key: value for key, value in self.event.raw_details.items() if key != "player"},
        )
        with self.assertRaisesRegex(ConversionError, "player_id"):
            EventToActionConverter().convert(self._inputs([missing]))

    @staticmethod
    def _event(
        *,
        event_id: str,
        index: int,
        event_type: str,
        source_type: str,
        timestamp: str,
        location: list[float],
        extra: dict[str, object] | None = None,
    ) -> CanonicalEvent:
        payload: dict[str, object] = {
            "id": event_id,
            "index": index,
            "period": 1,
            "timestamp": timestamp,
            "minute": 0,
            "second": int(float(timestamp.split(":")[-1])),
            "possession": 1,
            "possession_team": {"id": 10, "name": "Home"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 10, "name": "Home"},
            "player": {"id": 101, "name": "Player"},
            "type": {"id": 30, "name": source_type},
            "location": location,
            **(extra or {}),
        }
        endpoint = next(
            (
                value.get("end_location")
                for value in (extra or {}).values()
                if isinstance(value, dict) and value.get("end_location") is not None
            ),
            None,
        )
        return CanonicalEvent(
            source="statsbomb",
            source_event_id=UUID(event_id),
            source_record_index=index - 1,
            source_event_index=index,
            match_id=100,
            team_id=10,
            player_id=101,
            possession_id=1,
            possession_team_id=10,
            event_type=event_type,
            event_subtype=None,
            period=1,
            timestamp=time.fromisoformat(timestamp),
            minute=0,
            second=int(float(timestamp.split(":")[-1])),
            duration=0.5,
            x=location[0] / 120.0 * 100.0,
            y=location[1] / 80.0 * 100.0,
            end_x=(endpoint[0] / 120.0 * 100.0 if endpoint else None),
            end_y=(endpoint[1] / 80.0 * 100.0 if endpoint else None),
            outcome="complete" if event_type == "pass" else None,
            body_part="right_foot" if event_type == "pass" else None,
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
        home: dict[int, int] | None = None,
        away: dict[int, int] | None = None,
    ) -> SpadlInput:
        return SpadlInput(
            events=tuple(events),
            lineup_intervals=(),
            three_sixty_by_event={},
            home_team_by_match=home or {100: 10},
            away_team_by_match=away or {100: 20},
            report=SpadlInputValidationReport(
                match_count=len({event.match_id for event in events}),
                event_count=len(events),
                lineup_interval_count=0,
                three_sixty_count=0,
                events_with_subtype=0,
                issues=(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
