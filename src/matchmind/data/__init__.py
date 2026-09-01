"""Data ingestion utilities."""

from .normalizer import (
    Canonical360Frame,
    CanonicalEvent,
    CanonicalLineupInterval,
    NormalizationError,
    StatsBomb360Normalizer,
    StatsBombEventNormalizer,
    StatsBombLineupNormalizer,
)
from .ingestion import IngestionCounts, StatsBombIngestionService
from .postgres_writer import EventUpsertCounts, PostgresDataWriter
from .raw_reader import (
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
    "EventUpsertCounts",
    "IngestionCounts",
    "LineupIntervalValidationResult",
    "NormalizationError",
    "RawDataError",
    "RawDataFileNotFoundError",
    "RawDataFormatError",
    "RawMatchBundle",
    "RawRecord",
    "PostgresDataWriter",
    "StatsBombRawReader",
    "StatsBomb360Normalizer",
    "StatsBombEventNormalizer",
    "StatsBombLineupNormalizer",
    "StatsBombIngestionService",
    "ValidationIssue",
    "ValidationResult",
    "ThreeSixtyValidationResult",
]
