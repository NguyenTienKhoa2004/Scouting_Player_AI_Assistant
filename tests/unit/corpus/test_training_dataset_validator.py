from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.corpus.training_dataset_validator import (  # noqa: E402
    load_training_corpus_manifest,
)
from matchmind.model_dataset.training_manifest import (  # noqa: E402
    ProductionPromotionBlocked,
    require_corpus_adequate_for_promotion,
)


CORPUS_MANIFEST = (
    PROJECT_ROOT / "configs" / "datasets" / "vaep-training-corpus-v1.json"
)


class TrainingDatasetValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_training_corpus_manifest(CORPUS_MANIFEST)

    def test_pinned_corpus_reconciles_with_local_statsbomb_data(self) -> None:
        self.assertEqual(
            self.corpus.corpus_id,
            "statsbomb-open-data-male-multicompetition-v1",
        )
        self.assertEqual(len(self.corpus.matches), 1831)
        self.assertEqual(len(self.corpus.match_ids), 1831)
        self.assertEqual(len(self.corpus.selections), 10)
        self.assertEqual(self.corpus.scope["competition_gender"], "male")
        self.assertFalse(self.corpus.scope["three_sixty_required"])
        self.assertEqual(
            len({match.competition_id for match in self.corpus.matches}), 8
        )

    def test_complete_corpus_with_adequate_splits_passes_gate(self) -> None:
        assessment = self.corpus.assess(
            self.corpus.match_ids,
            self._split_manifest(matches=300, positives=500),
        )

        self.assertTrue(assessment["adequate_for_production_evaluation"])
        self.assertEqual(assessment["blockers"], [])
        self.assertEqual(assessment["materialized"]["coverage_rate"], 1.0)

    def test_world_cup_only_artifact_remains_experimental(self) -> None:
        world_cup_2022_ids = {
            match.match_id
            for match in self.corpus.matches
            if match.competition_id == 43 and match.season_id == 106
        }
        assessment = self.corpus.assess(
            world_cup_2022_ids,
            self._split_manifest(matches=10, positives=50),
        )

        self.assertFalse(assessment["adequate_for_production_evaluation"])
        self.assertEqual(assessment["materialized"]["match_count"], 64)
        self.assertIn("minimum_matches", assessment["blockers"])
        self.assertIn("complete_declared_corpus", assessment["blockers"])
        self.assertIn("test.minimum_concedes_positive", assessment["blockers"])

    def test_promotion_guard_rejects_inadequate_corpus(self) -> None:
        manifest = {
            "production_promotion": {
                "corpus_adequate": False,
                "blockers": ["corpus:minimum_matches"],
            }
        }

        with self.assertRaisesRegex(
            ProductionPromotionBlocked, "adequate materialized corpus"
        ):
            require_corpus_adequate_for_promotion(manifest)

    @staticmethod
    def _split_manifest(*, matches: int, positives: int) -> dict[str, object]:
        return {
            "splits": {
                split: {
                    "match_count": matches,
                    "scores_positive_count": positives,
                    "concedes_positive_count": positives,
                }
                for split in ("train", "validation", "test")
            }
        }


if __name__ == "__main__":
    unittest.main()
