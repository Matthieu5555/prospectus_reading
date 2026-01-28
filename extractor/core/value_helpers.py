"""Value extraction helpers for provenance-aware data handling.

These utilities handle both legacy format (plain values) and provenance
format ({"value": ..., "source_page": ...}) consistently.
"""

from typing import Any


def is_not_found(value: Any) -> bool:
    """Check if a value is NOT_FOUND in either legacy or provenance format.

    Handles three cases:
    1. None - explicitly missing
    2. "NOT_FOUND" - legacy string format
    3. {"value": "NOT_FOUND", ...} - provenance dict format
    """
    if value is None:
        return True
    if value == "NOT_FOUND":
        return True
    if isinstance(value, dict) and value.get("value") == "NOT_FOUND":
        return True
    return False


def get_raw_value(value: Any, default: Any = None) -> Any:
    """Extract raw value from either plain value or provenance dict.

    Handles both formats:
    - Plain: "Test Fund" -> "Test Fund"
    - Provenance: {"value": "Test Fund", "source_page": 1} -> "Test Fund"
    """
    if value is None:
        return default
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def is_actionable_not_found(value: Any) -> bool:
    """Check if a NOT_FOUND value needs investigation.

    A NOT_FOUND is actionable if it has no reason or reason is "extraction_failed".
    NOT_FOUND values with these reasons are NOT actionable:
    - "in_external_doc": Field is in KIID/Annual Report
    - "not_applicable": Field doesn't apply (e.g., Class X has no fees)
    - "not_in_document": Confirmed absent from document
    - "inherited": Value comes from umbrella level

    Args:
        value: The value to check (either plain or provenance dict format)

    Returns:
        True if this is a NOT_FOUND that needs investigation
    """
    if not is_not_found(value):
        return False

    # Plain format has no reason info - consider actionable
    if not isinstance(value, dict):
        return True

    # Check the reason
    reason = value.get("not_found_reason")
    if reason is None:
        return True  # No reason = needs investigation

    # These reasons mean we should NOT create a question
    non_actionable_reasons = {
        "in_external_doc",
        "not_applicable",
        "not_in_document",
        "inherited",
    }
    return reason not in non_actionable_reasons
