"""Load model-ready data for logistic-baseline training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from matchmind.vaep_features.feature_dataset_loader import baseline_feature_allowlist

from .logistic_contracts import PreparedLogisticData, TARGETS
from .training_files import find_training_file, read_json


def prepare_logistic_data(artifact_directory: Path) -> PreparedLogisticData:
    import pyarrow.parquet as pq

    root = Path(artifact_directory).resolve()
    manifest_path = root / "training_manifest.json"
    manifest = read_json(manifest_path)
    feature_manifest = read_json(root / "feature_allowlist.json")
    features = tuple(feature_manifest.get("feature_columns") or ())
    if features != baseline_feature_allowlist():
        raise ValueError("Dataset features do not match the expected VAEP feature list")

    dataset_path = find_training_file(root, manifest, "model_dataset.parquet")
    parquet_file = pq.ParquetFile(dataset_path)
    required_columns = {*features, *TARGETS, "split"}
    available_columns = set(parquet_file.schema_arrow.names)
    parquet_file.close()
    if not required_columns.issubset(available_columns):
        raise ValueError("Model dataset is missing required training columns")

    return PreparedLogisticData(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        feature_manifest=feature_manifest,
        features=features,
        dataset_path=dataset_path,
    )


def iter_split_batches(
    dataset_path: Path,
    features: tuple[str, ...],
    split: str,
    *,
    batch_size: int,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.dataset as ds

    columns = [*features, *TARGETS, "split"]
    scanner = ds.dataset(dataset_path, format="parquet").scanner(
        columns=columns,
        filter=ds.field("split") == split,
        batch_size=batch_size,
        use_threads=False,
    )
    for batch in scanner.to_batches():
        selected = pa.Table.from_batches([batch])
        if selected.num_rows == 0:
            continue
        matrix = np.column_stack(
            [selected[name].to_numpy(zero_copy_only=False) for name in features]
        ).astype(np.float32, copy=False)
        require_finite(matrix)
        targets = {
            target: selected[target]
            .to_numpy(zero_copy_only=False)
            .astype(np.int8, copy=False)
            for target in TARGETS
        }
        yield matrix, targets


def require_finite(matrix: Any) -> None:
    import numpy as np

    if not np.isfinite(matrix).all():
        raise ValueError("Model features contain missing or infinite values")


__all__ = ["iter_split_batches", "prepare_logistic_data", "require_finite"]
