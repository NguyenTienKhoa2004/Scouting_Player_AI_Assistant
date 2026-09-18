from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from matchmind.ingestion import IngestionCounts  # noqa: E402


class IngestionCountsTests(unittest.TestCase):
    def test_reconciles_raw_counts(self) -> None:
        counts = IngestionCounts(raw=100, accepted=80, rejected=5, deduplicated=15)
        self.assertTrue(counts.reconciled)

    def test_detects_unreconciled_counts(self) -> None:
        counts = IngestionCounts(raw=100, accepted=80, rejected=5, deduplicated=14)
        self.assertFalse(counts.reconciled)

    def test_reconciles_event_lineup_and_three_sixty_counts(self) -> None:
        counts = IngestionCounts(
            raw=10,
            accepted=10,
            raw_360=8,
            accepted_360=7,
            deduplicated_360=1,
            raw_lineup_intervals=4,
            accepted_lineup_intervals=4,
        )
        self.assertTrue(counts.reconciled)


if __name__ == "__main__":
    unittest.main()
