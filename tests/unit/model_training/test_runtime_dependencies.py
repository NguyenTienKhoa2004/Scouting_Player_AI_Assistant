from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.shared.runtime_dependencies import (  # noqa: E402
    RUNTIME_DEPENDENCY_FILENAME,
    RUNTIME_DEPENDENCY_MANIFEST_VERSION,
    capture_runtime_dependencies,
    write_runtime_dependency_manifest,
)


class RuntimeDependencyTests(unittest.TestCase):
    @staticmethod
    def _distribution(name: str, version: str) -> SimpleNamespace:
        return SimpleNamespace(metadata={"Name": name}, version=version)

    def test_ml_dependencies_are_exactly_pinned(self) -> None:
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("scikit-learn==1.9.1\n", requirements)
        self.assertIn("xgboost==3.4.1\n", requirements)

    @patch("matchmind.shared.runtime_dependencies.platform.machine", return_value="x86_64")
    @patch("matchmind.shared.runtime_dependencies.platform.release", return_value="test")
    @patch("matchmind.shared.runtime_dependencies.platform.system", return_value="TestOS")
    @patch(
        "matchmind.shared.runtime_dependencies.platform.python_version",
        return_value="3.12.0",
    )
    @patch(
        "matchmind.shared.runtime_dependencies.platform.python_implementation",
        return_value="CPython",
    )
    @patch(
        "matchmind.shared.runtime_dependencies.distribution_version",
        side_effect={
            "numpy": "2.3.3",
            "scikit-learn": "1.9.1",
            "xgboost": "3.4.1",
        }.get,
    )
    @patch("matchmind.shared.runtime_dependencies.distributions")
    def test_captures_every_installed_distribution_in_stable_order(
        self,
        installed_distributions,
        *_platform_mocks,
    ) -> None:
        installed_distributions.return_value = [
            SimpleNamespace(metadata={}, name=None),
            self._distribution("XGBoost", "3.4.1"),
            self._distribution("scikit_learn", "1.9.1"),
            self._distribution("NumPy", "2.3.3"),
        ]

        manifest = capture_runtime_dependencies().as_dict()

        self.assertEqual(
            list(manifest["packages"]),
            ["numpy", "scikit-learn", "xgboost"],
        )
        self.assertEqual(manifest["packages"]["scikit-learn"], "1.9.1")
        self.assertEqual(manifest["packages"]["xgboost"], "3.4.1")
        self.assertEqual(manifest["python"]["version"], "3.12.0")
        self.assertEqual(
            manifest["schema_version"], RUNTIME_DEPENDENCY_MANIFEST_VERSION
        )

    @patch("matchmind.shared.runtime_dependencies.capture_runtime_dependencies")
    def test_writes_dependency_manifest_inside_model_bundle(self, capture) -> None:
        capture.return_value = SimpleNamespace(
            as_dict=lambda: {
                "schema_version": RUNTIME_DEPENDENCY_MANIFEST_VERSION,
                "python": {"implementation": "CPython", "version": "3.12.0"},
                "platform": {
                    "system": "TestOS",
                    "release": "test",
                    "machine": "x86_64",
                },
                "packages": {
                    "scikit-learn": "1.9.1",
                    "xgboost": "3.4.1",
                },
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "models" / "score"
            target = write_runtime_dependency_manifest(bundle)
            manifest = json.loads(target.read_text(encoding="utf-8"))

            self.assertEqual(target, bundle / RUNTIME_DEPENDENCY_FILENAME)
            self.assertEqual(manifest["packages"]["scikit-learn"], "1.9.1")
            self.assertEqual(manifest["packages"]["xgboost"], "3.4.1")
            self.assertFalse(target.with_suffix(target.suffix + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
