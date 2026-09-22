from __future__ import annotations

import sys
import unittest
from collections.abc import Iterator
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.ingestion import (  # noqa: E402
    RawMatchBundle,
    RawRecord,
    build_ingestion_plan,
    fingerprint_match_bundle,
)


class FakeReader:
    def __init__(self, bundles: tuple[RawMatchBundle, ...]) -> None:
        self.bundles = bundles

    def iter_match_bundles(self) -> Iterator[RawMatchBundle]:
        yield from self.bundles


class IngestionPlannerTests(unittest.TestCase):
    def test_classifies_all_checkpoint_states_and_skips_unchanged(self) -> None:
        new_bundle = self._bundle(1001, "new")
        changed_bundle = self._bundle(1002, "changed-current")
        unchanged_bundle = self._bundle(1003, "same")
        unchanged_hash = fingerprint_match_bundle(unchanged_bundle).content_hash

        plan = build_ingestion_plan(
            FakeReader((new_bundle, changed_bundle, unchanged_bundle)),
            {
                1002: "b" * 64,
                1003: unchanged_hash,
                1004: "d" * 64,
            },
        )

        self.assertEqual([item.match_id for item in plan.new], [1001])
        self.assertEqual([item.match_id for item in plan.changed], [1002])
        self.assertEqual([item.match_id for item in plan.unchanged], [1003])
        self.assertEqual([item.match_id for item in plan.removed], [1004])
        self.assertEqual(plan.source_match_ids, (1001, 1002, 1003))
        self.assertEqual(plan.ingest_match_ids, (1001, 1002))
        self.assertEqual(
            tuple(item.match_id for item in plan.matches_for_ingestion(force=True)),
            (1001, 1002, 1003),
        )
        self.assertIsNone(plan.removed[0].content_hash)

    def test_rejects_duplicate_source_match_ids(self) -> None:
        reader = FakeReader((self._bundle(1001, "first"), self._bundle(1001, "second")))

        with self.assertRaisesRegex(ValueError, "duplicate match_id"):
            build_ingestion_plan(reader, {})

    @staticmethod
    def _bundle(match_id: int, marker: str) -> RawMatchBundle:
        return RawMatchBundle(
            match=RawRecord(
                kind="match",
                match_id=match_id,
                source_file=Path("matches/43/106.json"),
                source_record_index=0,
                payload={"match_id": match_id, "marker": marker},
            ),
            lineups=(),
            events=(),
            three_sixty=(),
        )


if __name__ == "__main__":
    unittest.main()
