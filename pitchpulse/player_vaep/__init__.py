"""Player-minute calculation and VAEP summary artifacts."""

from .artifacts import (
    PLAYER_VAEP_FILENAME,
    PLAYER_VAEP_MANIFEST_FILENAME,
    PlayerAggregationPaths,
    PlayerAggregationWriter,
)
from .calculations import (
    MINUTES_POLICY_VERSION,
    PLAYER_AGGREGATION_VERSION,
    MatchContext,
    PlayerAggregationError,
    PlayerMatchMinutes,
    PlayerVaepAggregator,
    calculate_player_minutes,
    match_duration_seconds,
)

__all__ = [
    "MINUTES_POLICY_VERSION",
    "PLAYER_AGGREGATION_VERSION",
    "PLAYER_VAEP_FILENAME",
    "PLAYER_VAEP_MANIFEST_FILENAME",
    "MatchContext",
    "PlayerAggregationError",
    "PlayerAggregationPaths",
    "PlayerAggregationWriter",
    "PlayerMatchMinutes",
    "PlayerVaepAggregator",
    "calculate_player_minutes",
    "match_duration_seconds",
]
