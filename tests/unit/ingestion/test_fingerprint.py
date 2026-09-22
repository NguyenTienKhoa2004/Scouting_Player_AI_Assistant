from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.ingestion import (  # noqa: E402
    RawMatchBundle,
    RawRecord,
    fingerprint_match_bundle,
)


class MatchFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_when_object_key_order_changes(self) -> None:
        original = self._bundle()
        reordered = self._bundle()
        reordered.match.payload.clear()
        reordered.match.payload.update(
            {
                "away_team": {"name": "Away", "id": 2},
                "home_team": {"name": "Home", "id": 1},
                "match_id": 1001,
            }
        )

        self.assertEqual(
            fingerprint_match_bundle(original),
            fingerprint_match_bundle(reordered),
        )

    def test_each_match_component_contributes_to_content_hash(self) -> None:
        original = self._bundle()
        baseline = fingerprint_match_bundle(original)

        mutations = (
            ("metadata", original.match.payload, "match_status", "available"),
            ("events", original.events[0].payload, "duration", 1.25),
            ("lineups", original.lineups[0].payload, "team_name", "Home"),
            ("three-sixty", original.three_sixty[0].payload, "visible_area", [1, 2]),
        )
        for label, _payload, key, value in mutations:
            with self.subTest(component=label):
                changed = copy.deepcopy(original)
                target = self._component_payload(changed, label)
                target[key] = value
                self.assertNotEqual(
                    baseline.content_hash,
                    fingerprint_match_bundle(changed).content_hash,
                )

    def test_event_array_order_changes_content_hash(self) -> None:
        original = self._bundle()
        second = RawRecord(
            kind="event",
            match_id=1001,
            source_file=Path("events/1001.json"),
            source_record_index=1,
            payload={"id": "event-2", "index": 2},
        )
        forward = RawMatchBundle(
            match=original.match,
            lineups=original.lineups,
            events=original.events + (second,),
            three_sixty=original.three_sixty,
        )
        reversed_events = RawMatchBundle(
            match=original.match,
            lineups=original.lineups,
            events=tuple(reversed(forward.events)),
            three_sixty=original.three_sixty,
        )

        self.assertNotEqual(
            fingerprint_match_bundle(forward).content_hash,
            fingerprint_match_bundle(reversed_events).content_hash,
        )

    @staticmethod
    def _component_payload(bundle: RawMatchBundle, label: str) -> dict[str, object]:
        if label == "metadata":
            return bundle.match.payload
        if label == "events":
            return bundle.events[0].payload
        if label == "lineups":
            return bundle.lineups[0].payload
        return bundle.three_sixty[0].payload

    @staticmethod
    def _bundle() -> RawMatchBundle:
        match_id = 1001
        return RawMatchBundle(
            match=RawRecord(
                kind="match",
                match_id=match_id,
                source_file=Path("matches/43/106.json"),
                source_record_index=0,
                payload={
                    "match_id": match_id,
                    "home_team": {"id": 1, "name": "Home"},
                    "away_team": {"id": 2, "name": "Away"},
                },
            ),
            lineups=(
                RawRecord(
                    kind="lineup",
                    match_id=match_id,
                    source_file=Path("lineups/1001.json"),
                    source_record_index=0,
                    payload={"team_id": 1, "lineup": []},
                ),
            ),
            events=(
                RawRecord(
                    kind="event",
                    match_id=match_id,
                    source_file=Path("events/1001.json"),
                    source_record_index=0,
                    payload={"id": "event-1", "index": 1},
                ),
            ),
            three_sixty=(
                RawRecord(
                    kind="three_sixty",
                    match_id=match_id,
                    source_file=Path("three-sixty/1001.json"),
                    source_record_index=0,
                    payload={"event_uuid": "event-1", "visible_area": []},
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
