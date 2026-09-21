from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pitchpulse.reproducibility.verification import (  # noqa: E402
    REQUIRED_REPRODUCIBLE_ARTIFACTS,
    ReproducibilityError,
    audit_training_artifacts,
    compare_independent_runs,
    finalize_reproducibility_report,
)


class ReproducibilityTests(unittest.TestCase):
    def test_independent_runs_match_and_receipt_is_added_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as raw:
            root = Path(raw)
            reference = root / "reference"
            candidate = root / "candidate"
            self._run(reference, path_marker="reference", latency=1.0)
            self._run(candidate, path_marker="candidate", latency=9.0)

            report = compare_independent_runs(reference, candidate)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["mismatches"], [])

            receipt = finalize_reproducibility_report(reference, report)
            manifest = json.loads(
                (reference / "training_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(receipt.is_file())
            self.assertEqual(manifest["reproducibility"]["status"], "passed")
            self.assertIn(receipt.name, manifest["artifacts"])

    def test_changed_artifact_fails_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as raw:
            root = Path(raw)
            reference = root / "reference"
            candidate = root / "candidate"
            self._run(reference, path_marker="reference", latency=1.0)
            self._run(candidate, path_marker="candidate", latency=2.0)
            changed = candidate / "model_dataset.parquet"
            changed.write_bytes(b"changed")
            self._refresh_hash(candidate, changed.name)

            report = compare_independent_runs(reference, candidate)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["mismatches"], [changed.name])
            with self.assertRaises(ReproducibilityError):
                finalize_reproducibility_report(reference, report)

    def test_manifest_must_be_hash_complete(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as raw:
            root = Path(raw) / "run"
            self._run(root, path_marker="run", latency=1.0)
            manifest_path = root / "training_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"].pop("action_values.parquet")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ReproducibilityError, "hash-complete"):
                audit_training_artifacts(root)

    @staticmethod
    def _run(root: Path, *, path_marker: str, latency: float) -> None:
        root.mkdir(parents=True)
        artifacts: dict[str, dict[str, str]] = {}
        for name in REQUIRED_REPRODUCIBLE_ARTIFACTS:
            path = root / name
            if name in {
                "runtime_dependencies.json",
                "xgboost_runtime_dependencies.json",
            }:
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "pitchpulse-runtime-dependencies-v1",
                            "python": {
                                "implementation": "CPython",
                                "version": "3.12.11",
                            },
                            "platform": {
                                "machine": "AMD64",
                                "release": "11",
                                "system": "Windows",
                            },
                            "packages": {
                                package: "1.0"
                                for package in (
                                    "joblib",
                                    "numpy",
                                    "pandas",
                                    "pyarrow",
                                    "scikit-learn",
                                    "scipy",
                                    "socceraction",
                                    "xgboost",
                                )
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            elif path.suffix == ".json" and name not in {
                "score_xgboost.json",
                "concede_xgboost.json",
            }:
                path.write_text(
                    json.dumps(
                        {
                            "result": "same",
                            "path": f"C:/{path_marker}/{name}",
                            "inference_latency": {"seconds": latency},
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                path.write_bytes(f"deterministic:{name}".encode("utf-8"))
            artifacts[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest = {
            "source": {
                "dataset_fingerprint": "a" * 64,
                "feature_manifest_sha256": "b" * 64,
            },
            "versions": {"feature": "v1", "split": "v1"},
            "split": {"assignment_fingerprint": "c" * 64},
            "artifacts": artifacts,
        }
        (root / "training_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    @staticmethod
    def _refresh_hash(root: Path, name: str) -> None:
        manifest_path = root / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = root / name
        manifest["artifacts"][name]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
