"""Pinned training-corpus selection and validation."""

from .match_metadata import (
    MatchMetadata,
    MatchMetadataError,
    load_statsbomb_match_metadata,
)
from .training_dataset_validator import (
    DEFAULT_ADEQUACY_POLICY,
    TRAINING_CORPUS_SCHEMA_VERSION,
    CorpusSelection,
    TrainingCorpus,
    TrainingCorpusError,
    load_training_corpus_manifest,
    single_file_training_corpus,
)

__all__ = [
    "CorpusSelection",
    "DEFAULT_ADEQUACY_POLICY",
    "MatchMetadata",
    "MatchMetadataError",
    "TRAINING_CORPUS_SCHEMA_VERSION",
    "TrainingCorpus",
    "TrainingCorpusError",
    "load_statsbomb_match_metadata",
    "load_training_corpus_manifest",
    "single_file_training_corpus",
]
