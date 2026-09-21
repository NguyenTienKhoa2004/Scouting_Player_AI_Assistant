from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.labeling_and_splitting.splits import (  # noqa: E402
    SPLIT_VERSION,
    ChronologicalMatchSplitter,
)
from pitchpulse.labeling_and_splitting.targets import TARGET_POLICY_VERSION  # noqa: E402
from pitchpulse.labeling_and_splitting.split_artifacts import (  # noqa: E402
    ChunkedSplitArtifactWriter,
)


class ChunkedSplitArtifactWriterTests(unittest.TestCase):
    def test_builds_complete_match_splits_and_streamed_label_counts(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            label_directory = root / "labels-test"
            label_directory.mkdir()
            matches_path = root / "matches.json"
            matches_path.write_text(
                json.dumps(
                    [
                        self._match(1, "2020-01-01"),
                        self._match(2, "2020-01-02"),
                        self._match(3, "2020-01-03"),
                    ]
                ),
                encoding="utf-8",
            )
            labels_path = label_directory / "action_labels.parquet"
            labels = pa.table(
                {
                    "match_id": pa.array([1, 1, 2, 2, 3, 3], pa.int64()),
                    "scores": pa.array([True, False, False, True, False, False]),
                    "concedes": pa.array(
                        [False, True, False, False, True, False]
                    ),
                    "eligible": pa.array([True] * 6),
                }
            )
            pq.write_table(labels.slice(0, 4), labels_path)
            with pq.ParquetWriter(
                label_directory / "combined.parquet", labels.schema
            ) as writer:
                writer.write_table(labels.slice(0, 4))
                writer.write_table(labels.slice(4, 2))
            (label_directory / "combined.parquet").replace(labels_path)
            label_manifest = {
                "source_dataset_fingerprint": "a" * 64,
                "target_policy_version": TARGET_POLICY_VERSION,
                "match_count": 3,
                "action_count": 6,
                "files": {
                    labels_path.name: self._sha256(labels_path),
                },
            }
            (label_directory / "label_manifest.json").write_text(
                json.dumps(label_manifest), encoding="utf-8"
            )

            build_manifest = ChronologicalMatchSplitter._manifest

            def marked_manifest(*args, **kwargs):
                manifest = build_manifest(*args, **kwargs)
                manifest["created_by_splitter"] = True
                return manifest

            with patch.object(
                ChronologicalMatchSplitter,
                "_manifest",
                side_effect=marked_manifest,
            ):
                paths = ChunkedSplitArtifactWriter().write(
                    label_directory,
                    match_metadata_path=matches_path,
                )
            assignments = pq.read_table(paths.assignments).to_pylist()
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

            self.assertEqual(
                [row["split"] for row in assignments],
                ["train", "validation", "test"],
            )
            self.assertEqual({row["split_version"] for row in assignments}, {SPLIT_VERSION})
            self.assertEqual(manifest["counts"], {"match_count": 3, "action_count": 6})
            self.assertEqual(manifest["splits"]["train"]["scores_positive_count"], 1)
            self.assertEqual(
                manifest["splits"]["validation"]["scores_positive_count"], 1
            )
            self.assertEqual(manifest["splits"]["test"]["concedes_positive_count"], 1)
            self.assertTrue(manifest["checks"]["match_level_isolation"])
            self.assertTrue(manifest["created_by_splitter"])

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
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
