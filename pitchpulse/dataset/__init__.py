"""Pinned training-dataset selection and validation."""

from .bronze import (
    BronzeSourceError,
    BronzeSourceReport,
    validate_bronze_repository,
)
from .match_metadata import (
    MatchMetadata,
    MatchMetadataError,
    load_statsbomb_match_metadata,
)
from .training_dataset_validator import (
    DEFAULT_ADEQUACY_POLICY,
    TRAINING_DATASET_SCHEMA_VERSION,
    BronzeContract,
    DatasetSelection,
    TrainingDataset,
    TrainingDatasetError,
    load_training_dataset_manifest,
    single_file_training_dataset,
)

__all__ = [
    "BronzeContract",
    "BronzeSourceError",
    "BronzeSourceReport",
    "DatasetSelection",
    "DEFAULT_ADEQUACY_POLICY",
    "MatchMetadata",
    "MatchMetadataError",
    "TRAINING_DATASET_SCHEMA_VERSION",
    "TrainingDataset",
    "TrainingDatasetError",
    "load_statsbomb_match_metadata",
    "load_training_dataset_manifest",
    "single_file_training_dataset",
    "validate_bronze_repository",
]
