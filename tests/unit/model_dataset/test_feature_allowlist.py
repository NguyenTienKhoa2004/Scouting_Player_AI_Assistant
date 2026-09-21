from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.model_dataset.feature_allowlist import (  # noqa: E402
    ModelDatasetError,
    validate_feature_allowlist,
)
from pitchpulse.vaep_features.feature_dataset_loader import (  # noqa: E402
    baseline_feature_allowlist,
)


class FeatureAllowlistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allowlist = baseline_feature_allowlist()

    def test_accepts_the_baseline_feature_allowlist(self) -> None:
        self.assertEqual(validate_feature_allowlist(self.allowlist), self.allowlist)

    def test_rejects_identifier_added_to_feature_allowlist(self) -> None:
        with self.assertRaisesRegex(ModelDatasetError, "exactly match"):
            validate_feature_allowlist(self.allowlist + ("match_id",))


if __name__ == "__main__":
    unittest.main()
