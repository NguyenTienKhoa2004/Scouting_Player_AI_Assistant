from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from socceraction.spadl import config as spadlconfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.labeling_and_splitting.label_artifacts import (  # noqa: E402
    ChunkedTargetLabelWriter,
)
from pitchpulse.spadl.converter import (  # noqa: E402
    ACTION_MAPPING_VERSION,
    COORDINATE_SYSTEM_VERSION,
)
from pitchpulse.vaep_features.action_state import STATE_CONTRACT_VERSION  # noqa: E402
from pitchpulse.vaep_features.feature_builder import BASE_FEATURE_VERSION  # noqa: E402


class ChunkedTargetLabelWriterTests(unittest.TestCase):
    def test_writes_one_label_row_group_per_action_row_group(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            feature_directory = root / "features"
            output_root = root / "models"
            feature_directory.mkdir()
            rows = [
                self._action(10, 0, team_id=1),
                self._action(10, 1, team_id=1, action_type="shot", result="success"),
                self._action(20, 0, team_id=2),
                self._action(20, 1, team_id=3),
            ]
            actions = pa.Table.from_pylist(rows)
            actions_path = feature_directory / "actions.parquet"
            with pq.ParquetWriter(actions_path, actions.schema) as writer:
                writer.write_table(actions.slice(0, 2))
                writer.write_table(actions.slice(2, 2))

            fingerprint = self._fingerprint(rows, event_count=4)
            quality = {
                "mapping_version": ACTION_MAPPING_VERSION,
                "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
                "state_contract_version": STATE_CONTRACT_VERSION,
                "feature_version": BASE_FEATURE_VERSION,
                "include_360": False,
                "event_count": 4,
                "action_count": 4,
            }
            self._json(feature_directory / "conversion_quality_report.json", quality)
            self._json(
                feature_directory / "manifest.json",
                {
                    **quality,
                    "analytics_run_id": 7,
                    "dataset_fingerprint": fingerprint,
                    "files": {
                        "actions.parquet": self._sha256(actions_path),
                    },
                },
            )
            matches_path = root / "matches.json"
            self._json(
                matches_path,
                [self._match(10, "2020-01-01"), self._match(20, "2020-01-02")],
            )

            result = ChunkedTargetLabelWriter().write(
                feature_directory,
                match_metadata_path=matches_path,
                output_root=output_root,
            )

            labels = pq.ParquetFile(result.labels)
            label_row_groups = labels.num_row_groups
            label_rows = labels.metadata.num_rows
            labels.close()
            manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertEqual(label_row_groups, 2)
            self.assertEqual(label_rows, 4)
            self.assertEqual(manifest["match_count"], 2)
            self.assertEqual(manifest["action_count"], 4)
            self.assertEqual(manifest["source_dataset_fingerprint"], fingerprint)
            self.assertEqual(
                manifest["files"]["action_labels.parquet"],
                self._sha256(result.labels),
            )

    @staticmethod
    def _action(
        match_id: int,
        action_id: int,
        *,
        team_id: int,
        action_type: str = "pass",
        result: str = "success",
    ) -> dict[str, object]:
        return {
            "match_id": match_id,
            "action_id": action_id,
            "period_id": 1,
            "team_id": team_id,
            "type_id": spadlconfig.actiontypes.index(action_type),
            "type_name": action_type,
            "result_id": spadlconfig.results.index(result),
            "result_name": result,
            "bodypart_id": spadlconfig.bodyparts.index("foot"),
            "is_shootout": False,
            "possession_changed": False,
            "source": "statsbomb",
            "source_event_id": f"{match_id}-{action_id}",
            "source_event_index": action_id,
            "source_action_index": action_id,
        }

    @staticmethod
    def _match(match_id: int, match_date: str) -> dict[str, object]:
        return {
            "match_id": match_id,
            "match_date": match_date,
            "kick_off": "12:00:00",
            "competition": {"competition_id": 1},
            "season": {"season_id": 1},
        }

    @staticmethod
    def _fingerprint(rows: list[dict[str, object]], *, event_count: int) -> str:
        digest = hashlib.sha256()
        digest.update(ACTION_MAPPING_VERSION.encode())
        digest.update(BASE_FEATURE_VERSION.encode())
        digest.update(str(event_count).encode())
        for row in rows:
            digest.update(
                (
                    f"{row['match_id']}|{row['source']}|{row['source_event_id']}|"
                    f"{row['source_event_index']}|{row['source_action_index']}\n"
                ).encode()
            )
        return digest.hexdigest()

    @staticmethod
    def _json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
