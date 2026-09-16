"""VAEP target generation and chronological match-level splitting."""

from .splits import (
    SPLIT_VERSION,
    ChronologicalMatchSplitter,
    SplitAssignmentError,
    SplitDataset,
    split_context,
)
from .targets import (
    TARGET_POLICY_VERSION,
    TargetDataset,
    TargetGenerationError,
    TargetLabelBuilder,
    baseline_target_policy,
)

__all__ = [
    "ChronologicalMatchSplitter",
    "SPLIT_VERSION",
    "SplitAssignmentError",
    "SplitDataset",
    "TARGET_POLICY_VERSION",
    "TargetDataset",
    "TargetGenerationError",
    "TargetLabelBuilder",
    "baseline_target_policy",
    "split_context",
]
