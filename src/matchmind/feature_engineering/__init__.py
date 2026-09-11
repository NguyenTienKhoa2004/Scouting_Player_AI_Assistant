"""StatsBomb-to-SPADL conversion and VAEP feature engineering."""

from .vaep_features import (
    BASE_FEATURE_VERSION,
    NB_PREV_ACTIONS,
    ActionFeatureBuilder,
    ActionFeatureRow,
    build_action_features,
)
from .action_state import (
    STATE_CONTRACT_VERSION,
    ActionState,
    ActionStateBuilder,
    StateConstructionError,
    build_action_states,
)
from .features_360 import (
    FEATURE_360_VERSION,
    ThreeSixtyFeatureEnricher,
    enrich_features_with_360,
)
from .pipeline import AnalyticsDataset, AnalyticsPipeline
from .input import (
    ContractIssue,
    SpadlInput,
    SpadlInputContractError,
    SpadlInputValidationReport,
    SpadlInputValidator,
)
from .spadl_converter import (
    ACTION_MAPPING_VERSION,
    COORDINATE_SYSTEM_VERSION,
    ConversionError,
    EventToActionConverter,
    SpadlAction,
    SpadlConversionReport,
    SpadlConversionResult,
    SOCCERACTION_VERSION,
    convert_events_to_actions,
)

__all__ = [
    "AnalyticsDataset",
    "AnalyticsPipeline",
    "ActionFeatureBuilder",
    "ActionFeatureRow",
    "ActionState",
    "ActionStateBuilder",
    "BASE_FEATURE_VERSION",
    "ContractIssue",
    "COORDINATE_SYSTEM_VERSION",
    "ACTION_MAPPING_VERSION",
    "ConversionError",
    "EventToActionConverter",
    "FEATURE_360_VERSION",
    "NB_PREV_ACTIONS",
    "SpadlInput",
    "SpadlInputContractError",
    "SpadlInputValidationReport",
    "SpadlInputValidator",
    "SpadlAction",
    "SpadlConversionReport",
    "SpadlConversionResult",
    "SOCCERACTION_VERSION",
    "STATE_CONTRACT_VERSION",
    "StateConstructionError",
    "ThreeSixtyFeatureEnricher",
    "convert_events_to_actions",
    "build_action_features",
    "build_action_states",
    "enrich_features_with_360",
]
