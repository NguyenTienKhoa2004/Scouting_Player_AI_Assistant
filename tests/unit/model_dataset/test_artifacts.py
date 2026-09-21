from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.vaep_features.feature_builder import BASE_FEATURE_VERSION  # noqa: E402
from pitchpulse.vaep_features.feature_dataset_loader import baseline_feature_allowlist  # noqa: E402
from pitchpulse.model_dataset.artifacts import (  # noqa: E402
    ChunkedModelDatasetWriter,
)
from pitchpulse.labeling_and_splitting.splits import SPLIT_VERSION  # noqa: E402
from pitchpulse.labeling_and_splitting.targets import TARGET_POLICY_VERSION  # noqa: E402


class ChunkedModelDatasetWriterTests(unittest.TestCase):
    def test_joins_aligned_row_groups_and_filters_ineligible_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as raw:
            root = Path(raw)
            feature_artifacts = root / "feature_artifacts"
            labels = root / "labels"
            feature_artifacts.mkdir()
            labels.mkdir()
            fingerprint = "a" * 64
            allowlist = baseline_feature_allowlist()
            feature_rows = {
                "match_id": [10, 10, 20, 30],
                "action_id": [0, 1, 0, 0],
                "feature_version": [BASE_FEATURE_VERSION] * 4,
            }
            for index, name in enumerate(allowlist):
                feature_rows[name] = [float(index), 1.0, 2.0, 3.0]
            features = pa.Table.from_pydict(feature_rows)
            self._write_groups(feature_artifacts / "action_features.parquet", features)
            self._json(
                feature_artifacts / "manifest.json",
                {
                    "dataset_fingerprint": fingerprint,
                    "files": {
                        "action_features.parquet": self._sha256(
                            feature_artifacts / "action_features.parquet"
                        )
                    },
                },
            )

            label_schema = pa.schema(
                [
                    pa.field("analytics_run_id", pa.int64(), nullable=True),
                    pa.field("match_id", pa.int64(), nullable=False),
                    pa.field("action_id", pa.int64(), nullable=False),
                    pa.field("scores", pa.bool_(), nullable=True),
                    pa.field("concedes", pa.bool_(), nullable=True),
                    pa.field("eligible", pa.bool_(), nullable=False),
                    pa.field("exclusion_reason", pa.string(), nullable=True),
                    pa.field("target_policy_version", pa.string(), nullable=False),
                ]
            )
            label_rows = [
                (7, 10, 0, False, False, True, None, TARGET_POLICY_VERSION),
                (7, 10, 1, None, None, False, "match_end", TARGET_POLICY_VERSION),
                (7, 20, 0, True, False, True, None, TARGET_POLICY_VERSION),
                (7, 30, 0, False, True, True, None, TARGET_POLICY_VERSION),
            ]
            label_table = pa.Table.from_pylist(
                [dict(zip(label_schema.names, row, strict=True)) for row in label_rows],
                schema=label_schema,
            )
            self._write_groups(labels / "action_labels.parquet", label_table)
            self._json(
                labels / "label_manifest.json",
                {
                    "source_dataset_fingerprint": fingerprint,
                    "files": {
                        "action_labels.parquet": self._sha256(
                            labels / "action_labels.parquet"
                        )
                    },
                },
            )
            assignments = pa.table(
                {
                    "match_id": pa.array([10, 20, 30], type=pa.int64()),
                    "competition_id": pa.array([1, 1, 1], type=pa.int64()),
                    "season_id": pa.array([1, 1, 1], type=pa.int64()),
                    "match_date": pa.array(
                        [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
                        type=pa.date32(),
                    ),
                    "split": ["train", "validation", "test"],
                    "split_order": pa.array([0, 1, 2], type=pa.int64()),
                    "split_version": [SPLIT_VERSION] * 3,
                }
            )
            pq.write_table(assignments, labels / "split_assignments.parquet")
            self._json(
                labels / "split_manifest.json",
                {
                    "source_dataset_fingerprint": fingerprint,
                    "files": {
                        "split_assignments.parquet": self._sha256(
                            labels / "split_assignments.parquet"
                        )
                    },
                    "splits": {
                        name: {"eligible_label_count": 1}
                        for name in ("train", "validation", "test")
                    },
                },
            )

            paths = ChunkedModelDatasetWriter().write(feature_artifacts, labels)
            result = pq.read_table(paths.model_dataset)

            self.assertEqual(result.num_rows, 3)
            self.assertEqual(result["match_id"].to_pylist(), [10, 20, 30])
            self.assertEqual(
                result["split"].to_pylist(), ["train", "validation", "test"]
            )
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["row_count"], 3)
            self.assertEqual(manifest["match_count"], 3)

    @staticmethod
    def _write_groups(path: Path, table: pa.Table) -> None:
        writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table.slice(0, 2))
        writer.write_table(table.slice(2, 2))
        writer.close()

    @staticmethod
    def _json(path: Path, value: dict[str, object]) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
