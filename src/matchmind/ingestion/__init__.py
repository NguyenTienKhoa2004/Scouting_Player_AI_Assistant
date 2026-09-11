"""StatsBomb reading, normalization, validation, and ingestion."""

from .normalizer import (
    Canonical360Frame,
    CanonicalEvent,
    CanonicalLineupInterval,
    NormalizationError,
    StatsBomb360Normalizer,
    StatsBombEventNormalizer,
    StatsBombLineupNormalizer,
)
from .pipeline import IngestionCounts, StatsBombIngestionService
from .reader import (
    RawDataError,
    RawDataFileNotFoundError,
    RawDataFormatError,
    RawMatchBundle,
    RawRecord,
    StatsBombRawReader,
)
from .validator import (
    CanonicalEventValidator,
    Canonical360Validator,
    CanonicalLineupIntervalValidator,
    EventValidationContext,
    LineupIntervalValidationResult,
    ThreeSixtyValidationResult,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    "CanonicalEvent",
    "Canonical360Frame",
    "Canonical360Validator",
    "CanonicalEventValidator",
    "CanonicalLineupInterval",
    "CanonicalLineupIntervalValidator",
    "EventValidationContext",
    "IngestionCounts",
    "LineupIntervalValidationResult",
    "NormalizationError",
    "RawDataError",
    "RawDataFileNotFoundError",
    "RawDataFormatError",
    "RawMatchBundle",
    "RawRecord",
    "StatsBombRawReader",
    "StatsBomb360Normalizer",
    "StatsBombEventNormalizer",
    "StatsBombLineupNormalizer",
    "StatsBombIngestionService",
    "ValidationIssue",
    "ValidationResult",
    "ThreeSixtyValidationResult",
]
