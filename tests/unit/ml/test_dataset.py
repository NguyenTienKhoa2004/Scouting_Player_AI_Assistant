from __future__ import annotations

import sys
import unittest
from datetime import date
from types import SimpleNamespace
from pathlib import Path

import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.analytics.features.vaep_features import BASE_FEATURE_VERSION  # noqa: E402
from matchmind.ml.feature_artifacts import baseline_feature_allowlist  # noqa: E402
from matchmind.ml.dataset import (  # noqa: E402
    MODEL_DATASET_VERSION,
    ModelDatasetBuilder,
    ModelDatasetError,
    select_model_matrix,
)
from matchmind.ml.splits import SPLIT_VERSION  # noqa: E402
from matchmind.ml.targets import TARGET_POLICY_VERSION  # noqa: E402


class ModelDatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allowlist = baseline_feature_allowlist()
        self.features = self._features()
        self.labels = self._labels()
        self.splits = self._splits()
        self.feature_artifacts = SimpleNamespace(
            features=self.features,
            feature_allowlist=self.allowlist,
            lineage=SimpleNamespace(
                analytics_run_id=7,
                dataset_fingerprint="a" * 64,
                feature_version=BASE_FEATURE_VERSION,
            ),
        )

    def test_joins_only_eligible_rows_and_builds_allowlisted_matrix(self) -> None:
        result = ModelDatasetBuilder().build(
            self.feature_artifacts, self.labels, self.splits
        )

        self.assertEqual(result.table.num_rows, 3)
        self.assertEqual(result.table["action_id"].to_pylist(), [0, 0, 1])
        self.assertEqual(
            result.table["split"].to_pylist(), ["train", "validation", "validation"]
        )
        self.assertEqual(
            result.table.schema.metadata[b"model_dataset_version"].decode(),
            MODEL_DATASET_VERSION,
        )

        matrix = result.model_matrix()
        self.assertEqual(tuple(matrix.column_names), self.allowlist)
        self.assertFalse(
            {"analytics_run_id", "match_id", "action_id", "scores", "concedes",
             "split", "feature_version", "target_policy_version", "split_version"}
            & set(matrix.column_names)
        )
        self.assertEqual(result.model_matrix(split="train").num_rows, 1)
        self.assertEqual(
            result.allowlist_manifest["selection_policy"],
            "explicit_allowlist_only",
        )
        self.assertFalse(
            result.allowlist_manifest["checks"][
                "identifiers_or_metadata_in_model_matrix"
            ]
        )

    def test_extra_dataset_columns_still_cannot_enter_model_matrix(self) -> None:
        result = ModelDatasetBuilder().build(
            self.feature_artifacts, self.labels, self.splits
        )
        tampered = result.table.append_column(
            "source_event_id", pa.array(["leak-1", "leak-2", "leak-3"])
        )

        matrix = select_model_matrix(
            tampered,
            feature_allowlist=self.allowlist,
        )

        self.assertNotIn("source_event_id", matrix.column_names)
        self.assertEqual(tuple(matrix.column_names), self.allowlist)

    def test_rejects_identifier_added_to_feature_allowlist(self) -> None:
        with self.assertRaisesRegex(ModelDatasetError, "exactly match"):
            select_model_matrix(
                self.features,
                feature_allowlist=self.allowlist + ("match_id",),
            )

    def test_rejects_non_reconciling_label_keys(self) -> None:
        labels = self.labels.set_column(
            self.labels.schema.get_field_index("action_id"),
            "action_id",
            pa.array([0, 2, 0, 1], type=pa.int64()),
        )

        with self.assertRaisesRegex(ModelDatasetError, "do not reconcile"):
            ModelDatasetBuilder().build(self.feature_artifacts, labels, self.splits)

    def test_rejects_duplicate_match_split_assignment(self) -> None:
        duplicate = pa.concat_tables([self.splits, self.splits.slice(0, 1)])

        with self.assertRaisesRegex(ModelDatasetError, "duplicate match_id"):
            ModelDatasetBuilder().build(self.feature_artifacts, self.labels, duplicate)

    def _features(self) -> pa.Table:
        rows: dict[str, list[object]] = {
            "match_id": [10, 10, 20, 20],
            "action_id": [0, 1, 0, 1],
            "feature_version": [BASE_FEATURE_VERSION] * 4,
        }
        for index, name in enumerate(self.allowlist):
            rows[name] = [float(index), 0.0, 1.0, 2.0]
        return pa.Table.from_pydict(rows)

    @staticmethod
    def _labels() -> pa.Table:
        schema = pa.schema(
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
        return pa.Table.from_pylist(
            [
                {
                    "analytics_run_id": 7,
                    "match_id": 10,
                    "action_id": 0,
                    "scores": False,
                    "concedes": False,
                    "eligible": True,
                    "exclusion_reason": None,
                    "target_policy_version": TARGET_POLICY_VERSION,
                },
                {
                    "analytics_run_id": 7,
                    "match_id": 10,
                    "action_id": 1,
                    "scores": None,
                    "concedes": None,
                    "eligible": False,
                    "exclusion_reason": "penalty_shootout",
                    "target_policy_version": TARGET_POLICY_VERSION,
                },
                {
                    "analytics_run_id": 7,
                    "match_id": 20,
                    "action_id": 0,
                    "scores": True,
                    "concedes": False,
                    "eligible": True,
                    "exclusion_reason": None,
                    "target_policy_version": TARGET_POLICY_VERSION,
                },
                {
                    "analytics_run_id": 7,
                    "match_id": 20,
                    "action_id": 1,
                    "scores": False,
                    "concedes": True,
                    "eligible": True,
                    "exclusion_reason": None,
                    "target_policy_version": TARGET_POLICY_VERSION,
                },
            ],
            schema=schema,
        )

    @staticmethod
    def _splits() -> pa.Table:
        schema = pa.schema(
            [
                pa.field("match_id", pa.int64(), nullable=False),
                pa.field("competition_id", pa.int64(), nullable=False),
                pa.field("season_id", pa.int64(), nullable=False),
                pa.field("match_date", pa.date32(), nullable=False),
                pa.field("split", pa.string(), nullable=False),
                pa.field("split_order", pa.int64(), nullable=False),
                pa.field("split_version", pa.string(), nullable=False),
            ]
        )
        return pa.Table.from_pylist(
            [
                {
                    "match_id": 10,
                    "competition_id": 43,
                    "season_id": 106,
                    "match_date": date(2022, 11, 20),
                    "split": "train",
                    "split_order": 0,
                    "split_version": SPLIT_VERSION,
                },
                {
                    "match_id": 20,
                    "competition_id": 43,
                    "season_id": 106,
                    "match_date": date(2022, 11, 21),
                    "split": "validation",
                    "split_order": 1,
                    "split_version": SPLIT_VERSION,
                },
            ],
            schema=schema,
        )


if __name__ == "__main__":
    unittest.main()
