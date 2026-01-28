"""Converters for transforming LLM output to typed ExtractedValue models.

These functions handle the conversion from various input formats
(legacy strings, provenance dicts, ExtractedValue objects) to
properly typed ExtractedValue instances.
"""

from extractor.pydantic_models import ExtractedValue, NOT_FOUND, NotFoundReason


def not_found_reason_from_string(reason_str: str | None) -> NotFoundReason:
    """Parse NotFoundReason from string, defaulting to EXTRACTION_FAILED.

    Args:
        reason_str: String representation of the reason, e.g., "not_in_document"

    Returns:
        Corresponding NotFoundReason enum value
    """
    if not reason_str:
        return NotFoundReason.EXTRACTION_FAILED

    reason_map = {
        "not_in_document": NotFoundReason.NOT_IN_DOCUMENT,
        "not_applicable": NotFoundReason.NOT_APPLICABLE,
        "in_external_doc": NotFoundReason.IN_EXTERNAL_DOC,
        "inherited": NotFoundReason.INHERITED,
        "extraction_failed": NotFoundReason.EXTRACTION_FAILED,
    }
    return reason_map.get(reason_str.lower(), NotFoundReason.EXTRACTION_FAILED)


def to_name_extracted_value(data: dict | str | None, default: str = "Unknown") -> ExtractedValue | str:
    """Convert name data to ExtractedValue or string, preserving provenance.

    Names are special because they should never be NOT_FOUND - we always
    need some identifier. This function ensures we get a usable value.

    Args:
        data: Input in any supported format (ExtractedValue, dict, str, None)
        default: Fallback value if input is None or NOT_FOUND

    Returns:
        ExtractedValue with the name, or the default wrapped in ExtractedValue

    Handles:
        - ExtractedValue: return as-is
        - Provenance dict: convert to ExtractedValue
        - Simple string: wrap in ExtractedValue
        - None: use default wrapped in ExtractedValue
    """
    if data is None:
        return ExtractedValue.from_simple(default)

    # Already an ExtractedValue
    if isinstance(data, ExtractedValue):
        return data

    # Provenance format (dict with "value" key)
    if isinstance(data, dict) and "value" in data:
        value = data.get("value")
        if value == NOT_FOUND or not value:
            return ExtractedValue.from_simple(default)
        return ExtractedValue(
            value=value,
            source_page=data.get("source_page"),
            source_quote=data.get("source_quote"),
            rationale=data.get("rationale"),
            confidence=data.get("confidence", 1.0),
        )

    # Simple string
    if isinstance(data, str):
        if data == NOT_FOUND or not data:
            return ExtractedValue.from_simple(default)
        return ExtractedValue.from_simple(data)

    return ExtractedValue.from_simple(default)


def to_extracted_value(data: dict | str | bool | int | float | None, field_name: str = "") -> ExtractedValue | None:
    """Convert LLM output to ExtractedValue.

    This is the main conversion function for field values. It handles both
    provenance format (dict with "value" key) and legacy format (plain values).

    Args:
        data: Input value in any supported format
        field_name: Name of the field being converted (for error messages)

    Returns:
        ExtractedValue instance, or None if input is None

    Handles:
        - None: returns None
        - Provenance dict: full conversion with all metadata
        - Legacy primitives (str, bool, int, float): wrapped in ExtractedValue
        - NOT_FOUND string: converted to extraction_failed ExtractedValue
    """
    if data is None:
        return None

    # Provenance format (new) - dict with "value" key
    if isinstance(data, dict) and "value" in data:
        value = data.get("value")
        reason = None
        external_ref = None

        # Parse NOT_FOUND reason if present
        if value == NOT_FOUND:
            reason = not_found_reason_from_string(data.get("not_found_reason"))
            external_ref = data.get("external_reference")

        return ExtractedValue(
            value=value,
            source_page=data.get("source_page"),
            source_quote=data.get("source_quote"),
            rationale=data.get("rationale"),
            confidence=data.get("confidence", 1.0),
            not_found_reason=reason,
            external_reference=external_ref,
        )

    # Legacy format (simple value)
    if isinstance(data, (str, bool, int, float)):
        if data == NOT_FOUND:
            return ExtractedValue.extraction_failed(f"Legacy extraction - {field_name}")
        return ExtractedValue.from_simple(data)

    # Fallback for unexpected types - convert to string
    return ExtractedValue.from_simple(str(data) if data else NOT_FOUND)
