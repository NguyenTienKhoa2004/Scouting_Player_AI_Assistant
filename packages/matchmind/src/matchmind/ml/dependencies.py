"""Capture the exact runtime environment alongside every model bundle."""

from __future__ import annotations

import json
import platform
import re
from dataclasses import dataclass
from importlib.metadata import distributions, version as distribution_version
from pathlib import Path
from typing import Any


RUNTIME_DEPENDENCY_MANIFEST_VERSION = "matchmind-runtime-dependencies-v1"
RUNTIME_DEPENDENCY_FILENAME = "runtime_dependencies.json"


def _canonical_distribution_name(name: str) -> str:
    """Apply the package-name normalization defined by Python packaging."""

    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True, slots=True)
class RuntimeDependencySnapshot:
    """Serializable record of the interpreter and installed distributions."""

    python_implementation: str
    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    packages: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_DEPENDENCY_MANIFEST_VERSION,
            "python": {
                "implementation": self.python_implementation,
                "version": self.python_version,
            },
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
                "machine": self.platform_machine,
            },
            "packages": dict(self.packages),
        }


def capture_runtime_dependencies() -> RuntimeDependencySnapshot:
    """Capture exact versions for every distribution in the training runtime."""

    package_names: set[str] = set()
    for distribution in distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raw_name = getattr(distribution, "name", None)
        if not raw_name:
            # Broken third-party ``.dist-info`` directories occasionally have no
            # Name metadata. They cannot be resolved reproducibly and must not
            # prevent the named training dependencies from being recorded.
            continue
        package_names.add(_canonical_distribution_name(str(raw_name)))

    # Resolve by package name so duplicate metadata directories cannot make the
    # result depend on discovery order. This matches what imports see on sys.path.
    package_versions = {
        name: distribution_version(name) for name in sorted(package_names)
    }

    return RuntimeDependencySnapshot(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        packages=tuple(sorted(package_versions.items())),
    )


def write_runtime_dependency_manifest(bundle_directory: Path) -> Path:
    """Write an atomic, deterministic dependency manifest into a model bundle."""

    bundle_directory.mkdir(parents=True, exist_ok=True)
    target = bundle_directory / RUNTIME_DEPENDENCY_FILENAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            capture_runtime_dependencies().as_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


__all__ = [
    "RUNTIME_DEPENDENCY_FILENAME",
    "RUNTIME_DEPENDENCY_MANIFEST_VERSION",
    "RuntimeDependencySnapshot",
    "capture_runtime_dependencies",
    "write_runtime_dependency_manifest",
]
