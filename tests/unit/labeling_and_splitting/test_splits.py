from __future__ import annotations

import sys
import unittest
from datetime import date, time
from pathlib import Path

import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.corpus.match_metadata import MatchMetadata  # noqa: E402
from matchmind.labeling_and_splitting.splits import (  # noqa: E402
    SPLIT_VERSION,
    ChronologicalMatchSplitter,
    SplitAssignmentError,
)


class ChronologicalMatchSplitterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = pa.Table.from_pylist(
            [
                {"match_id": match_id, "action_id": action_id}
                for match_id in range(1, 7)
                for action_id in range(2)
            ]
        )
        self.matches = tuple(
            MatchMetadata(
                match_id=match_id,
                competition_id=43,
                season_id=106,
                match_date=date(2022, 11, match_id),
                kick_off=time(12),
            )
            for match_id in reversed(range(1, 7))
        )

    def test_assigns_complete_matches_chronologically(self) -> None:
        result = self._build()
        rows = result.assignments.to_pylist()

        self.assertEqual([row["match_id"] for row in rows], [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            [row["split"] for row in rows],
            ["train", "train", "train", "train", "validation", "test"],
        )
        self.assertTrue(result.manifest["checks"]["match_level_isolation"])
        self.assertFalse(result.manifest["policy"]["uses_vaep_fit"])

    def test_regeneration_is_identical(self) -> None:
        first = self._build()
        second = self._build()

        self.assertTrue(first.assignments.equals(second.assignments))
        self.assertEqual(
            first.manifest["assignment_fingerprint"],
            second.manifest["assignment_fingerprint"],
        )
        self.assertEqual(
            set(first.assignments["split_version"].to_pylist()), {SPLIT_VERSION}
        )

    def test_rejects_missing_match_metadata(self) -> None:
        with self.assertRaisesRegex(SplitAssignmentError, "missing action matches"):
            ChronologicalMatchSplitter().build(
                self.actions,
                self.matches[:-1],
                dataset_fingerprint="a" * 64,
                target_policy_version="target-v1",
            )

    def _build(self):
        return ChronologicalMatchSplitter().build(
            self.actions,
            self.matches,
            dataset_fingerprint="a" * 64,
            target_policy_version="target-v1",
        )


if __name__ == "__main__":
    unittest.main()
