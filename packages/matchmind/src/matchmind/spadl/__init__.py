"""Canonical-event validation and conversion to SPADL actions."""

from .converter import (
    ACTION_MAPPING_VERSION,
    COORDINATE_SYSTEM_VERSION,
    SOCCERACTION_VERSION,
    ConversionError,
    EventToActionConverter,
    SpadlAction,
    SpadlConversionReport,
    SpadlConversionResult,
    convert_events_to_actions,
)
from .input_reader import SPADL_INPUT_SCHEMA, PostgresSpadlInputReader
from .input_validator import (
    ContractIssue,
    SpadlInput,
    SpadlInputContractError,
    SpadlInputValidationReport,
    validate_spadl_input,
)

__all__ = [
    "ACTION_MAPPING_VERSION",
    "COORDINATE_SYSTEM_VERSION",
    "ContractIssue",
    "ConversionError",
    "EventToActionConverter",
    "PostgresSpadlInputReader",
    "SOCCERACTION_VERSION",
    "SPADL_INPUT_SCHEMA",
    "SpadlAction",
    "SpadlConversionReport",
    "SpadlConversionResult",
    "SpadlInput",
    "SpadlInputContractError",
    "SpadlInputValidationReport",
    "convert_events_to_actions",
    "validate_spadl_input",
]
