from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.player_vaep.artifacts import (  # noqa: E402
    PlayerAggregationWriter,
)
from pitchpulse.shared.file_io import file_sha256, read_json, write_json  # noqa: E402


class PlayerAggregationWriterTests(unittest.TestCase):
    def test_writes_player_artifact_and_updates_training_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_manifest = self._write_dataset(root)
            actions_path = root / "actions.parquet"
            values_path = root / "action_values.parquet"
            self._write_actions(actions_path)
            self._write_action_values(values_path)
            action_manifest_path = root / "action_values_manifest.json"
            write_json(
                action_manifest_path,
                {
                    "dataset_fingerprint": "fingerprint",
                    "source_artifacts": {
                        "actions.parquet": {
                            "path": str(actions_path),
                            "sha256": file_sha256(actions_path),
                        }
                    },
                },
            )
            write_json(
                root / "training_manifest.json",
                {
                    "status": "action_valuation_complete",
                    "stage": "action_valuation",
                    "action_valuation": {"row_count": 3, "match_count": 1},
                    "artifacts": {
                        "action_values.parquet": {
                            "path": str(values_path),
                            "sha256": file_sha256(values_path),
                        },
                        "action_values_manifest.json": {
                            "path": str(action_manifest_path),
                            "sha256": file_sha256(action_manifest_path),
                        },
                    },
                },
            )

            paths = PlayerAggregationWriter().write(
                root, dataset_manifest, minimum_minutes=60
            )

            self.assertTrue(paths.player_vaep.is_file())
            output = pq.read_table(paths.player_vaep).to_pandas()
            self.assertEqual(len(output), 8)
            season_total = output[
                (output["aggregation_level"] == "competition_season")
                & (output["action_type"] == "all")
            ]
            self.assertEqual(int(season_total["action_count"].sum()), 2)
            self.assertAlmostEqual(float(season_total["player_vaep"].sum()), 0.3)
            output_manifest = read_json(paths.manifest)
            self.assertTrue(output_manifest["audit"]["reconciled"])
            self.assertEqual(output_manifest["audit"]["unknown_player_action_count"], 1)
            training_manifest = read_json(paths.training_manifest)
            self.assertEqual(training_manifest["status"], "player_aggregation_complete")

            repeated = PlayerAggregationWriter().write(
                root, dataset_manifest, minimum_minutes=60
            )
            self.assertEqual(repeated, paths)

    @staticmethod
    def _write_dataset(root: Path) -> Path:
        data = root / "bronze"
        (data / "matches" / "1").mkdir(parents=True)
        (data / "lineups").mkdir()
        (data / "events").mkdir()
        (data / "matches" / "1" / "2.json").write_text(
            json.dumps([{"match_id": 100}]), encoding="utf-8"
        )
        (data / "lineups" / "100.json").write_text(
            json.dumps(
                [
                    {
                        "team_id": 10,
                        "lineup": [
                            {
                                "player_id": 1,
                                "positions": [
                                    {
                                        "position": "Midfield",
                                        "from": "00:00",
                                        "to": None,
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "team_id": 20,
                        "lineup": [
                            {
                                "player_id": 2,
                                "positions": [
                                    {
                                        "position": "Forward",
                                        "from": "60:00",
                                        "to": None,
                                    }
                                ],
                            }
                        ],
                    },
                ]
            ),
            encoding="utf-8",
        )
        (data / "events" / "100.json").write_text(
            json.dumps([{"period": 2, "minute": 95, "second": 0}]),
            encoding="utf-8",
        )
        manifest = root / "dataset.json"
        write_json(
            manifest,
            {
                "bronze": {"data_root": str(data)},
                "selections": [
                    {
                        "competition_id": 1,
                        "season_id": 2,
                        "matches_path": "matches/1/2.json",
                    }
                ],
            },
        )
        return manifest

    @staticmethod
    def _write_actions(path: Path) -> None:
        pq.write_table(
            pa.table(
                {
                    "match_id": [100, 100, 100],
                    "action_id": [0, 1, 2],
                    "type_name": ["pass", "shot", "pass"],
                }
            ),
            path,
        )

    @staticmethod
    def _write_action_values(path: Path) -> None:
        pq.write_table(
            pa.table(
                {
                    "match_id": [100, 100, 100],
                    "action_id": [0, 1, 2],
                    "player_id": pa.array([1, 2, None], type=pa.int64()),
                    "team_id": [10, 20, 20],
                    "offensive_value": [0.1, 0.2, 0.4],
                    "defensive_value": [0.0, 0.0, 0.0],
                    "vaep_value": [0.1, 0.2, 0.4],
                }
            ),
            path,
        )


if __name__ == "__main__":
    unittest.main()
