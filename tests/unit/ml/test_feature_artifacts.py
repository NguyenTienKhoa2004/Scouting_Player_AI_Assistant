from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.analytics.features import (  # noqa: E402
    ACTION_MAPPING_VERSION,
    BASE_FEATURE_VERSION,
    COORDINATE_SYSTEM_VERSION,
    STATE_CONTRACT_VERSION,
)
from matchmind.ml import (  # noqa: E402
    FeatureArtifactValidationError,
    FeatureArtifactLoader,
    baseline_feature_allowlist,
)


class FeatureArtifactLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.artifact = self._write_artifact()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_a_hash_verified_reconciled_baseline(self) -> None:
        loaded = FeatureArtifactLoader().load(
            self.artifact,
            expected_analytics_run_id=7,
        )

        self.assertEqual(loaded.actions.num_rows, 2)
        self.assertEqual(loaded.features.num_rows, 2)
        self.assertEqual(loaded.lineage.analytics_run_id, 7)
        self.assertEqual(len(loaded.feature_allowlist), 568)
        self.assertFalse(any(name.endswith("_a3") for name in loaded.feature_allowlist))
        for suffix in ("_a0", "_a1", "_a2"):
            self.assertTrue(
                any(name.endswith(suffix) for name in loaded.feature_allowlist)
            )

    def test_rejects_a_file_whose_hash_changed(self) -> None:
        with (self.artifact / "action_features.parquet").open("ab") as target:
            target.write(b"tampered")

        with self.assertRaisesRegex(
            FeatureArtifactValidationError, "SHA-256 mismatch"
        ):
            FeatureArtifactLoader().load(self.artifact)

    def test_rejects_wrong_state_lineage(self) -> None:
        quality_path = self.artifact / "conversion_quality_report.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality["state_contract_version"] = "wrong-state-v1"
        self._write_json(quality_path, quality)
        self._refresh_hash("conversion_quality_report.json")

        with self.assertRaisesRegex(
            FeatureArtifactValidationError, "state_contract_version"
        ):
            FeatureArtifactLoader().load(self.artifact)

    def test_rejects_requested_lineage_that_does_not_match(self) -> None:
        loader = FeatureArtifactLoader()

        with self.assertRaisesRegex(FeatureArtifactValidationError, "analytics run"):
            loader.load(self.artifact, expected_analytics_run_id=8)
        with self.assertRaisesRegex(
            FeatureArtifactValidationError, "dataset fingerprint"
        ):
            loader.load(self.artifact, expected_dataset_fingerprint="0" * 64)

    def test_rejects_non_reconciling_action_and_feature_keys(self) -> None:
        path = self.artifact / "action_features.parquet"
        table = pq.read_table(path)
        table = table.set_column(
            table.schema.get_field_index("action_id"),
            "action_id",
            pa.array([0, 2], type=pa.int64()),
        )
        pq.write_table(table, path)
        self._refresh_hash("action_features.parquet")

        with self.assertRaisesRegex(FeatureArtifactValidationError, "do not reconcile"):
            FeatureArtifactLoader().load(self.artifact)

    def test_rejects_any_column_outside_exact_feature_allowlist(self) -> None:
        path = self.artifact / "action_features.parquet"
        table = pq.read_table(path).append_column(
            "future_leak_a3", pa.array([False, False])
        )
        pq.write_table(table, path)
        self._refresh_hash("action_features.parquet")

        with self.assertRaisesRegex(
            FeatureArtifactValidationError, "exact baseline allowlist"
        ):
            FeatureArtifactLoader().load(self.artifact)

    def test_recomputes_and_rejects_a_false_dataset_fingerprint(self) -> None:
        path = self.artifact / "actions.parquet"
        table = pq.read_table(path)
        table = table.set_column(
            table.schema.get_field_index("source_event_index"),
            "source_event_index",
            pa.array([99, 100], type=pa.int64()),
        )
        pq.write_table(table, path)
        self._refresh_hash("actions.parquet")

        with self.assertRaisesRegex(
            FeatureArtifactValidationError, "fingerprint does not reconcile"
        ):
            FeatureArtifactLoader().load(self.artifact)

    def _write_artifact(self) -> Path:
        action_rows = {
            "match_id": [101, 101],
            "action_id": [0, 1],
            "source": ["statsbomb", "statsbomb"],
            "source_event_id": ["event-1", "event-2"],
            "source_event_index": [0, 1],
            "source_action_index": [0, 0],
            "mapping_version": [ACTION_MAPPING_VERSION] * 2,
            "coordinate_system_version": [COORDINATE_SYSTEM_VERSION] * 2,
            "state_version": [STATE_CONTRACT_VERSION] * 2,
            "period_id": [1, 1],
            "time_seconds": [1.0, 2.0],
            "team_id": [10, 10],
            "player_id": [1001, 1002],
            "type_id": [0, 11],
            "type_name": ["pass", "shot"],
            "result_id": [1, 1],
            "result_name": ["success", "success"],
            "is_shootout": [False, False],
        }
        event_count = 2
        fingerprint = self._fingerprint(action_rows, event_count)
        artifact = self.root / f"dataset-{fingerprint[:16]}"
        artifact.mkdir()

        metadata = {
            b"mapping_version": ACTION_MAPPING_VERSION.encode(),
            b"coordinate_system_version": COORDINATE_SYSTEM_VERSION.encode(),
            b"state_contract_version": STATE_CONTRACT_VERSION.encode(),
            b"feature_version": BASE_FEATURE_VERSION.encode(),
            b"dataset_fingerprint": fingerprint.encode(),
        }
        action_table = pa.Table.from_pydict(action_rows).replace_schema_metadata(
            metadata
        )
        pq.write_table(action_table, artifact / "actions.parquet")

        feature_rows: dict[str, list[object]] = {
            "match_id": [101, 101],
            "action_id": [0, 1],
            "feature_version": [BASE_FEATURE_VERSION] * 2,
        }
        for name in baseline_feature_allowlist():
            feature_rows[name] = [0.0, 0.0]
        feature_table = pa.Table.from_pydict(feature_rows).replace_schema_metadata(
            metadata
        )
        pq.write_table(feature_table, artifact / "action_features.parquet")

        quality_report = {
            "action_count": 2,
            "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
            "event_count": event_count,
            "feature_count": 2,
            "feature_version": BASE_FEATURE_VERSION,
            "include_360": False,
            "mapping_version": ACTION_MAPPING_VERSION,
            "match_count": 1,
            "state_contract_version": STATE_CONTRACT_VERSION,
            "state_count": 2,
        }
        self._write_json(
            artifact / "conversion_quality_report.json", quality_report
        )
        manifest = {
            "analytics_run_id": 7,
            "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
            "dataset_fingerprint": fingerprint,
            "feature_version": BASE_FEATURE_VERSION,
            "include_360": False,
            "mapping_version": ACTION_MAPPING_VERSION,
            "state_contract_version": STATE_CONTRACT_VERSION,
            "files": {
                filename: self._sha256(artifact / filename)
                for filename in (
                    "actions.parquet",
                    "action_features.parquet",
                    "conversion_quality_report.json",
                )
            },
        }
        self._write_json(artifact / "manifest.json", manifest)
        return artifact

    def _refresh_hash(self, filename: str) -> None:
        manifest_path = self.artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][filename] = self._sha256(self.artifact / filename)
        self._write_json(manifest_path, manifest)

    @staticmethod
    def _fingerprint(action_rows: dict[str, list[object]], event_count: int) -> str:
        digest = hashlib.sha256()
        digest.update(ACTION_MAPPING_VERSION.encode())
        digest.update(BASE_FEATURE_VERSION.encode())
        digest.update(str(event_count).encode())
        columns = [
            action_rows[name]
            for name in (
                "match_id",
                "source",
                "source_event_id",
                "source_event_index",
                "source_action_index",
            )
        ]
        for values in zip(*columns, strict=True):
            digest.update(("|".join(str(value) for value in values) + "\n").encode())
        return digest.hexdigest()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
