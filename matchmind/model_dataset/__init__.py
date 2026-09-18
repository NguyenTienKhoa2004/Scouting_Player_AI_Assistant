"""Join verified features, labels, and splits into model-ready datasets."""

from .builder import ModelDataset, ModelDatasetBuilder
from .feature_allowlist import (
    FEATURE_ALLOWLIST_FILENAME,
    MODEL_DATASET_FILENAME,
    MODEL_DATASET_SCHEMA_VERSION,
    MODEL_DATASET_VERSION,
    ModelDatasetError,
    feature_allowlist_manifest,
    select_model_matrix,
    validate_feature_allowlist,
)
from .artifacts import (
    MODEL_DATASET_MANIFEST_FILENAME,
    ChunkedModelDatasetPaths,
    ChunkedModelDatasetWriter,
)
from .finalize import (
    ChunkedPreparationFinalizer,
    FinalizedPreparationPaths,
)
from .training_manifest import PREPARATION_VERSION

__all__ = [
    "ChunkedModelDatasetPaths",
    "ChunkedModelDatasetWriter",
    "ChunkedPreparationFinalizer",
    "FEATURE_ALLOWLIST_FILENAME",
    "FinalizedPreparationPaths",
    "MODEL_DATASET_MANIFEST_FILENAME",
    "MODEL_DATASET_FILENAME",
    "MODEL_DATASET_SCHEMA_VERSION",
    "MODEL_DATASET_VERSION",
    "ModelDataset",
    "ModelDatasetBuilder",
    "ModelDatasetError",
    "PREPARATION_VERSION",
    "feature_allowlist_manifest",
    "select_model_matrix",
    "validate_feature_allowlist",
]
