"""Dataset loading and leakage-safe validation partitioning for XGBoost."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from matchmind.vaep_features.feature_dataset_loader import baseline_feature_allowlist

from .xgboost_contracts import (
    LoadedSplit,
    PreparedXGBoostData,
    TARGETS,
    XGBoostTrainingError,
)
from .training_files import find_training_file, read_json


def prepare_training_data(
    artifact_directory: Path,
    *,
    batch_size: int,
    progress: Any | None = None,
) -> PreparedXGBoostData:
    """Load the approved train/validation data and create validation phases."""

    import numpy as np

    root = Path(artifact_directory).resolve()
    manifest_path = root / "training_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("models", {}).get("status") != "logistic_baselines_evaluated":
        raise XGBoostTrainingError(
            "Run logistic baseline training before XGBoost training"
        )

    feature_manifest = read_json(root / "feature_allowlist.json")
    features = tuple(feature_manifest.get("feature_columns") or ())
    if features != baseline_feature_allowlist():
        raise XGBoostTrainingError(
            "Dataset features do not match the expected VAEP feature list"
        )

    dataset_path = find_training_file(root, manifest, "model_dataset.parquet")
    split_path = find_training_file(
        root, manifest, "split_assignments.parquet"
    )
    if progress is not None:
        progress("loading train rows as a sparse CSR matrix")
    train = load_split(dataset_path, features, "train", batch_size=batch_size)
    if progress is not None:
        progress("loading validation rows without touching test")
    validation = load_split(
        dataset_path,
        features,
        "validation",
        batch_size=batch_size,
    )

    phase_matches = validation_phase_matches(split_path, validation.match_ids)
    phase_indices = {
        phase: np.flatnonzero(np.isin(validation.match_ids, list(match_ids)))
        for phase, match_ids in phase_matches.items()
    }
    validate_phase_targets(validation.targets, phase_indices)
    return PreparedXGBoostData(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        feature_manifest=feature_manifest,
        features=features,
        train=train,
        validation=validation,
        phase_matches=phase_matches,
        phase_indices=phase_indices,
    )


def load_split(
    dataset_path: Path,
    features: tuple[str, ...],
    split: str,
    *,
    batch_size: int,
) -> LoadedSplit:
    import numpy as np
    import pyarrow as pa
    import pyarrow.dataset as ds
    from scipy import sparse

    if split not in {"train", "validation"}:
        raise XGBoostTrainingError(
            "XGBoost training may load only train and validation splits"
        )
    columns = ["match_id", *features, *TARGETS, "split"]
    scanner = ds.dataset(dataset_path, format="parquet").scanner(
        columns=columns,
        filter=ds.field("split") == split,
        batch_size=batch_size,
        use_threads=False,
    )
    matrices: list[Any] = []
    targets: dict[str, list[Any]] = {target: [] for target in TARGETS}
    match_ids: list[Any] = []
    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])
        if table.num_rows == 0:
            continue
        dense = np.column_stack(
            [table[name].to_numpy(zero_copy_only=False) for name in features]
        ).astype(np.float32, copy=False)
        if not np.isfinite(dense).all():
            raise XGBoostTrainingError(
                f"{split} model features contain null or non-finite values"
            )
        matrix = sparse.csr_matrix(dense)
        matrix.eliminate_zeros()
        matrices.append(matrix)
        match_ids.append(
            table["match_id"]
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        for target in TARGETS:
            targets[target].append(
                table[target]
                .to_numpy(zero_copy_only=False)
                .astype(np.int8, copy=False)
            )
    if not matrices:
        raise XGBoostTrainingError(f"{split} split contains no rows")
    return LoadedSplit(
        matrix=sparse.vstack(matrices, format="csr", dtype=np.float32),
        targets={target: np.concatenate(values) for target, values in targets.items()},
        match_ids=np.concatenate(match_ids),
    )


def validation_phase_matches(
    split_path: Path, observed_match_ids: Any
) -> dict[str, set[int]]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    assignments = pq.read_table(
        split_path,
        columns=["match_id", "split", "split_order"],
    )
    validation = assignments.filter(pc.equal(assignments["split"], "validation"))
    rows = sorted(validation.to_pylist(), key=lambda row: row["split_order"])
    ordered = [int(row["match_id"]) for row in rows]
    observed = {int(value) for value in set(observed_match_ids.tolist())}
    if len(ordered) < 3:
        raise XGBoostTrainingError(
            "validation requires at least three complete matches for tuning and "
            "calibration"
        )
    if len(ordered) != len(set(ordered)) or set(ordered) != observed:
        raise XGBoostTrainingError(
            "validation match IDs do not reconcile with split assignments"
        )
    tune_end = max(1, math.floor(len(ordered) * 0.50))
    calibration_fit_end = max(tune_end + 1, math.floor(len(ordered) * 0.75))
    calibration_fit_end = min(calibration_fit_end, len(ordered) - 1)
    return {
        "tuning": set(ordered[:tune_end]),
        "calibration_fit": set(ordered[tune_end:calibration_fit_end]),
        "calibration_evaluation": set(ordered[calibration_fit_end:]),
    }


def validate_phase_targets(
    targets: Mapping[str, Any], phase_indices: Mapping[str, Any]
) -> None:
    import numpy as np

    for phase, indices in phase_indices.items():
        if not len(indices):
            raise XGBoostTrainingError(f"validation phase {phase} has no rows")
        for target in TARGETS:
            if set(np.unique(targets[target][indices])) != {0, 1}:
                raise XGBoostTrainingError(
                    f"validation phase {phase} requires both classes for {target}"
                )


def split_summary(split: LoadedSplit) -> dict[str, Any]:
    return {
        "match_count": int(len(set(split.match_ids.tolist()))),
        "row_count": int(split.matrix.shape[0]),
        "feature_count": int(split.matrix.shape[1]),
        "nonzero_count": int(split.matrix.nnz),
        "matrix_density": float(
            split.matrix.nnz / (split.matrix.shape[0] * split.matrix.shape[1])
        ),
        "targets": {
            target: {
                "positive_count": int(split.targets[target].sum()),
                "positive_rate": float(split.targets[target].mean()),
            }
            for target in TARGETS
        },
    }


__all__ = [
    "load_split",
    "prepare_training_data",
    "split_summary",
    "validate_phase_targets",
    "validation_phase_matches",
]
