from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.ingestion import (  # noqa: E402
    RawDataFileNotFoundError,
    RawDataFormatError,
    StatsBombRawReader,
)


class StatsBombRawReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)
        (self.data_root / "matches" / "43").mkdir(parents=True)
        (self.data_root / "events").mkdir()
        (self.data_root / "lineups").mkdir()
        (self.data_root / "three-sixty").mkdir()

        self._write_json(
            self.data_root / "matches" / "43" / "106.json",
            [{"match_id": 1001, "home_team": {}, "away_team": {}}],
        )
        self._write_json(
            self.data_root / "events" / "1001.json",
            [{"id": "event-1", "type": {"name": "Pass"}}],
        )
        self._write_json(
            self.data_root / "lineups" / "1001.json",
            [{"team_id": 1}, {"team_id": 2}],
        )
        self._write_json(
            self.data_root / "three-sixty" / "1001.json",
            [{"event_uuid": "event-1", "visible_area": [], "freeze_frame": []}],
        )
        self.reader = StatsBombRawReader(self.data_root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reads_one_match_bundle_without_mutating_payload(self) -> None:
        bundle = self.reader.read_match_bundle(1001)

        self.assertEqual(bundle.match.match_id, 1001)
        self.assertEqual(len(bundle.lineups), 2)
        self.assertEqual(len(bundle.events), 1)
        self.assertEqual(len(bundle.three_sixty), 1)
        self.assertEqual(bundle.events[0].source_record_index, 0)
        self.assertEqual(bundle.events[0].payload["id"], "event-1")
        self.assertNotIn("match_id", bundle.events[0].payload)
        self.assertEqual(bundle.three_sixty[0].kind, "three_sixty")

    def test_missing_optional_three_sixty_file_returns_empty(self) -> None:
        (self.data_root / "three-sixty" / "1001.json").unlink()
        self.assertEqual(self.reader.read_three_sixty(1001), ())
        with self.assertRaisesRegex(RawDataFileNotFoundError, "1001.json"):
            self.reader.read_three_sixty(1001, required=True)

    def test_iter_events_uses_selected_match_ids(self) -> None:
        records = list(self.reader.iter_events([1001]))
        self.assertEqual([record.payload["id"] for record in records], ["event-1"])

    def test_missing_event_file_has_contextual_error(self) -> None:
        with self.assertRaisesRegex(RawDataFileNotFoundError, "1002.json"):
            self.reader.read_events(1002)

    def test_rejects_non_array_source_file(self) -> None:
        self._write_json(self.data_root / "events" / "1001.json", {"id": "event-1"})
        with self.assertRaisesRegex(RawDataFormatError, "Expected a JSON array"):
            self.reader.read_events(1001)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
