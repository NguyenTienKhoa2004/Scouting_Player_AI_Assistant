"""Reproducible model training and VAEP valuation support."""

from .feature_artifacts import (
    FeatureArtifactValidationError,
    FeatureArtifactLoader,
    FeatureArtifacts,
    FeatureArtifactLineage,
    baseline_feature_allowlist,
)
from .dependencies import (
    RUNTIME_DEPENDENCY_MANIFEST_VERSION,
    RuntimeDependencySnapshot,
    capture_runtime_dependencies,
    write_runtime_dependency_manifest,
)
from .corpus import (
    DEFAULT_ADEQUACY_POLICY,
    TRAINING_CORPUS_SCHEMA_VERSION,
    CorpusSelection,
    TrainingCorpus,
    TrainingCorpusError,
    load_training_corpus_manifest,
    single_file_training_corpus,
)
from .dataset import (
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
from .label_artifacts import (
    LABEL_MANIFEST_FILENAME,
    ChunkedTargetLabelWriter,
    LabelArtifactPaths,
)
from .model_dataset_artifacts import (
    MODEL_DATASET_MANIFEST_FILENAME,
    ChunkedModelDatasetPaths,
    ChunkedModelDatasetWriter,
)
from .finalize_preparation import (
    ChunkedPreparationFinalizer,
    FinalizedPreparationPaths,
)
from .evaluation import binary_probability_report
from .logistic_baseline import (
    LOGISTIC_BASELINE_REPORT_FILENAME,
    LOGISTIC_BASELINE_VERSION,
    PREPROCESSING_VERSION,
    LogisticBaselinePaths,
    LogisticBaselineTrainer,
)
from .vaep_data_preparation import (
    PREPARATION_VERSION,
    PreparationArtifactPaths,
    VaepDataPreparationWriter,
)
from .splits import (
    SPLIT_VERSION,
    ChronologicalMatchSplitter,
    MatchMetadata,
    SplitAssignmentError,
    load_statsbomb_match_metadata,
)
from .split_artifacts import ChunkedSplitArtifactWriter, SplitArtifactPaths
from .targets import (
    TARGET_POLICY_VERSION,
    TargetGenerationError,
    TargetLabelBuilder,
    baseline_target_policy,
)
from .training_manifest import (
    TRAINING_MANIFEST_FILENAME,
    TRAINING_MANIFEST_VERSION,
    ProductionPromotionBlocked,
    build_preparation_training_manifest,
    require_corpus_adequate_for_promotion,
)

__all__ = [
    "FeatureArtifactValidationError",
    "FeatureArtifactLoader",
    "FeatureArtifacts",
    "FeatureArtifactLineage",
    "CorpusSelection",
    "DEFAULT_ADEQUACY_POLICY",
    "FEATURE_ALLOWLIST_FILENAME",
    "LABEL_MANIFEST_FILENAME",
    "MODEL_DATASET_FILENAME",
    "MODEL_DATASET_MANIFEST_FILENAME",
    "MODEL_DATASET_SCHEMA_VERSION",
    "MODEL_DATASET_VERSION",
    "ModelDataset",
    "ModelDatasetBuilder",
    "ModelDatasetError",
    "PREPARATION_VERSION",
    "RUNTIME_DEPENDENCY_MANIFEST_VERSION",
    "SPLIT_VERSION",
    "TARGET_POLICY_VERSION",
    "TRAINING_CORPUS_SCHEMA_VERSION",
    "TRAINING_MANIFEST_FILENAME",
    "TRAINING_MANIFEST_VERSION",
    "ChronologicalMatchSplitter",
    "ChunkedTargetLabelWriter",
    "ChunkedSplitArtifactWriter",
    "ChunkedModelDatasetPaths",
    "ChunkedModelDatasetWriter",
    "ChunkedPreparationFinalizer",
    "FinalizedPreparationPaths",
    "LOGISTIC_BASELINE_REPORT_FILENAME",
    "LOGISTIC_BASELINE_VERSION",
    "PREPROCESSING_VERSION",
    "LogisticBaselinePaths",
    "LogisticBaselineTrainer",
    "LabelArtifactPaths",
    "MatchMetadata",
    "PreparationArtifactPaths",
    "ProductionPromotionBlocked",
    "RuntimeDependencySnapshot",
    "SplitAssignmentError",
    "SplitArtifactPaths",
    "TargetGenerationError",
    "TargetLabelBuilder",
    "TrainingCorpus",
    "TrainingCorpusError",
    "VaepDataPreparationWriter",
    "baseline_feature_allowlist",
    "baseline_target_policy",
    "binary_probability_report",
    "build_preparation_training_manifest",
    "capture_runtime_dependencies",
    "feature_allowlist_manifest",
    "load_statsbomb_match_metadata",
    "load_training_corpus_manifest",
    "require_corpus_adequate_for_promotion",
    "select_model_matrix",
    "single_file_training_corpus",
    "validate_feature_allowlist",
    "write_runtime_dependency_manifest",
]
