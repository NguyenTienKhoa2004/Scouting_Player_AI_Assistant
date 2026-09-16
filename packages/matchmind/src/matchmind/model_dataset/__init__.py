"""Join verified features, labels, and splits into model-ready datasets."""

from .builder import (
    FEATURE_ALLOWLIST_FILENAME,
    MODEL_DATASET_FILENAME,
    MODEL_DATASET_SCHEMA_VERSION,
    MODEL_DATASET_VERSION,
    ModelDataset,
    ModelDatasetBuilder,
    ModelDatasetError,
    feature_allowlist_manifest,
    select_model_matrix,
    validate_feature_allowlist,
)
from .preparation import (
    PREPARATION_VERSION,
    PreparationArtifactPaths,
    VaepDataPreparationWriter,
)

__all__ = [
    "FEATURE_ALLOWLIST_FILENAME",
    "MODEL_DATASET_FILENAME",
    "MODEL_DATASET_SCHEMA_VERSION",
    "MODEL_DATASET_VERSION",
    "ModelDataset",
    "ModelDatasetBuilder",
    "ModelDatasetError",
    "PREPARATION_VERSION",
    "PreparationArtifactPaths",
    "VaepDataPreparationWriter",
    "feature_allowlist_manifest",
    "select_model_matrix",
    "validate_feature_allowlist",
]
