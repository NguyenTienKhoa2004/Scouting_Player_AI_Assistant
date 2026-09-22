from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.dataset.bronze import BronzeSourceError  # noqa: E402
from pitchpulse.dataset.training_dataset_validator import (  # noqa: E402
    DEFAULT_ADEQUACY_POLICY,
    load_training_dataset_manifest,
)
from pitchpulse.dataset.update_bronze import (  # noqa: E402
    build_updated_manifest,
    update_bronze_repository,
)


class BronzeUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temporary_directory.name)
        self.configs = self.root / "configs"
        self.repository = self.root / "statsbomb-open-data"
        self.data_root = self.repository / "data"
        self.configs.mkdir()
        (self.data_root / "matches" / "1").mkdir(parents=True)
        (self.data_root / "events").mkdir()
        (self.data_root / "lineups").mkdir()
        (self.data_root / "three-sixty").mkdir()
        (self.repository / "LICENSE.pdf").write_bytes(b"fixture-license")
        self.matches_path = self.data_root / "matches" / "1" / "1.json"
        self._write_matches(match_ids=(1001,))
        (self.data_root / "events" / "1001.json").write_text("[]", encoding="utf-8")
        (self.data_root / "lineups" / "1001.json").write_text("[]", encoding="utf-8")
        self.source_manifest = self.configs / "dataset-v1.json"
        self.manifest = self._manifest()
        self.source_manifest.write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reconciles_commit_selection_hash_counts_and_summary(self) -> None:
        (self.data_root / "events" / "1002.json").write_text("[]", encoding="utf-8")
        (self.data_root / "lineups" / "1002.json").write_text("[]", encoding="utf-8")
        self._write_matches(match_ids=(1001, 1002))

        updated = build_updated_manifest(
            self.manifest,
            manifest_path=self.source_manifest,
            commit="b" * 40,
            dataset_id="fixture-v2",
        )

        self.assertEqual(updated["dataset_id"], "fixture-v2")
        self.assertEqual(updated["source"]["git_commit"], "b" * 40)
        self.assertEqual(updated["selections"][0]["expected_matches"], 2)
        self.assertEqual(
            updated["selections"][0]["sha256"],
            hashlib.sha256(self.matches_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(updated["expected_summary"]["match_count"], 2)

    def test_update_requires_explicit_commit_and_creates_new_manifest(self) -> None:
        with self.assertRaisesRegex(BronzeSourceError, "exact 40-character"):
            update_bronze_repository(
                source_manifest=self.source_manifest,
                output_manifest=self.configs / "bad.json",
                commit="latest",
                dataset_id="fixture-v2",
                fetch=False,
            )

        commands: list[tuple[str, ...]] = []

        def fake_git(repository_root: Path, *arguments: str) -> str:
            self.assertEqual(repository_root, self.repository)
            commands.append(arguments)
            if arguments == ("rev-parse", "HEAD"):
                return "a" * 40
            return ""

        output = self.configs / "dataset-v2.json"
        with (
            patch("pitchpulse.dataset.update_bronze._run_git", side_effect=fake_git),
            patch("pitchpulse.dataset.update_bronze.validate_bronze_repository"),
        ):
            result = update_bronze_repository(
                source_manifest=self.source_manifest,
                output_manifest=output,
                commit="b" * 40,
                dataset_id="fixture-v2",
                fetch=False,
            )

        self.assertEqual(result, output)
        self.assertIn(("cat-file", "-e", f"{'b' * 40}^{{commit}}"), commands)
        self.assertIn(("checkout", "--detach", "b" * 40), commands)
        dataset = load_training_dataset_manifest(output)
        self.assertEqual(dataset.dataset_id, "fixture-v2")
        self.assertEqual(dataset.source["git_commit"], "b" * 40)

    def _write_matches(self, *, match_ids: tuple[int, ...]) -> None:
        rows = [
            {
                "match_id": match_id,
                "competition": {
                    "competition_id": 1,
                    "competition_name": "Fixture League",
                },
                "season": {"season_id": 1, "season_name": "2025/2026"},
                "match_date": f"2026-01-{index:02d}",
                "kick_off": "12:00:00",
            }
            for index, match_id in enumerate(match_ids, start=1)
        ]
        self.matches_path.write_text(json.dumps(rows), encoding="utf-8")

    def _manifest(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "dataset_id": "fixture-v1",
            "description": "fixture",
            "source": {
                "name": "StatsBomb Open Data",
                "repository": "https://github.com/statsbomb/open-data",
                "git_commit": "a" * 40,
                "license_file": "../statsbomb-open-data/LICENSE.pdf",
                "attribution_required": True,
            },
            "bronze": {
                "layer": "bronze",
                "data_root": "../statsbomb-open-data/data",
                "source_format": "statsbomb-open-data-json",
                "immutable": True,
                "required_families": ["matches", "events", "lineups"],
                "optional_families": ["three-sixty"],
            },
            "scope": {
                "competition_gender": "male",
                "three_sixty_required": False,
                "event_and_lineup_files_required": True,
            },
            "adequacy_policy": dict(DEFAULT_ADEQUACY_POLICY),
            "expected_summary": {
                "match_count": 1,
                "competition_count": 1,
                "competition_season_count": 1,
                "first_match_date": "2026-01-01",
                "last_match_date": "2026-01-01",
            },
            "selections": [
                {
                    "competition_id": 1,
                    "competition_name": "Fixture League",
                    "season_id": 1,
                    "season_name": "2025/2026",
                    "matches_path": "matches/1/1.json",
                    "expected_matches": 1,
                    "sha256": hashlib.sha256(
                        self.matches_path.read_bytes()
                    ).hexdigest(),
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
