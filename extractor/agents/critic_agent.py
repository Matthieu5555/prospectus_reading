"""Critic agent — a second LLM pass that catches hallucinations and typos.

A second pass matters because LLMs can confidently fabricate values that look
plausible but don't appear anywhere in the source text.  The critic re-reads
the same pages the reader saw and checks every extracted value against the
original text, flagging mismatches, wrong page references, and invented
numbers.  It also assigns a per-field confidence score so downstream code
knows which values to trust and which need human review.
"""

import copy
import json

from extractor.core.config import DEFAULT_MODELS, ConfidenceThresholds
from extractor.core.cost_tracker import CostTracker
from extractor.core.llm_client import LLMClient
from extractor.core.value_helpers import get_raw_value
from extractor.pydantic_models.pipeline import CriticResult, FieldVerification
from extractor.prompts.critic_prompt import CRITIC_SYSTEM_PROMPT, build_critic_prompt


async def run_critic(
    entity_type: str,
    entity_name: str,
    extracted_data: dict,
    source_text: str,
    model: str = DEFAULT_MODELS["critic"],
    cost_tracker: CostTracker | None = None,
    parsed_tables: dict | None = None,
) -> CriticResult:
    """Run the critic to verify an extraction.

    Args:
        entity_type: Type of entity (umbrella, subfund, shareclass).
        entity_name: Name of the entity.
        extracted_data: The extracted data to verify.
        source_text: Source document text.
        model: LLM model to use (needs reasoning).
        cost_tracker: Optional tracker to record token usage.
        parsed_tables: Optional dict of table_type -> ParsedTable for ground truth.

    Returns:
        CriticResult with verification details.
    """
    client = LLMClient(cost_tracker=cost_tracker)

    try:
        return await client.complete_structured(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=build_critic_prompt(
                entity_type,
                entity_name,
                json.dumps(extracted_data, indent=2),
                source_text,
                parsed_tables=parsed_tables,
            ),
            model=model,
            response_model=CriticResult,
            agent="critic",
        )
    except Exception as e:
        # Return low confidence result on any error
        return CriticResult(
            entity_type=entity_type,
            entity_name=entity_name,
            verifications=[],
            overall_confidence=ConfidenceThresholds.FALLBACK,
            critical_errors=[f"Critic error: {e}"],
        )


async def verify_and_correct(
    entity_type: str,
    entity_name: str,
    extracted_data: dict,
    source_text: str,
    model: str = DEFAULT_MODELS["critic"],
    cost_tracker: CostTracker | None = None,
    parsed_tables: dict | None = None,
) -> tuple[dict, CriticResult]:
    """Verify extraction and apply corrections.

    Args:
        entity_type: Type of entity.
        entity_name: Name of the entity.
        extracted_data: The extracted data.
        source_text: Source document text.
        model: LLM model for critic.
        cost_tracker: Optional tracker to record token usage.
        parsed_tables: Optional dict of table_type -> ParsedTable for ground truth.

    Returns:
        Tuple of (corrected_data, critic_result).
    """
    critic_result = await run_critic(
        entity_type, entity_name, extracted_data, source_text, model, cost_tracker,
        parsed_tables=parsed_tables,
    )

    # Apply corrections (deepcopy to avoid mutating original)
    corrected = copy.deepcopy(extracted_data)
    for verification in critic_result.verifications:
        if verification.correction and verification.confidence in ["not_found", "low"]:
            # Apply correction
            if verification.field_name in corrected:
                corrected[verification.field_name] = verification.correction

    return corrected, critic_result


def compute_confidence_score(verifications: list[FieldVerification]) -> float:
    """Compute overall confidence from field verifications.

    Args:
        verifications: List of field verifications.

    Returns:
        Confidence score 0.0 to 1.0.
    """
    if not verifications:
        return ConfidenceThresholds.FALLBACK

    confidence_values = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.3,
        "not_found": 0.0,
    }

    scores = [
        confidence_values.get(verification.confidence, ConfidenceThresholds.FALLBACK)
        for verification in verifications
    ]
    return sum(scores) / len(scores)


async def verify_fund_extraction(
    fund_name: str,
    fund_data: dict,
    source_text: str,
    share_classes: list[dict] | None = None,
    model: str = DEFAULT_MODELS["critic"],
    cost_tracker: CostTracker | None = None,
    parsed_tables: dict | None = None,
) -> tuple[dict, list[CriticResult]]:
    """Verify a complete fund extraction including share classes.

    Args:
        fund_name: Name of the fund.
        fund_data: Extracted fund data.
        source_text: Source document text.
        share_classes: Extracted share classes (optional).
        model: LLM model for critic.
        cost_tracker: Optional tracker to record token usage.
        parsed_tables: Optional dict of table_type -> ParsedTable for ground truth.
                       Tables (ISIN, fee) are authoritative for share class verification.

    Returns:
        Tuple of (corrected_fund_data, list of critic results).
    """
    results = []

    # Verify fund-level data (no tables needed at fund level)
    fund_corrected, fund_result = await verify_and_correct(
        "subfund", fund_name, fund_data, source_text, model, cost_tracker
    )
    results.append(fund_result)

    # Verify each share class (pass tables for ISIN/fee ground truth)
    corrected_classes = []
    if share_classes:
        for sc in share_classes:
            sc_name = get_raw_value(sc.get("name"), "Unknown")
            sc_corrected, sc_result = await verify_and_correct(
                "shareclass", f"{fund_name}/{sc_name}", sc, source_text, model, cost_tracker,
                parsed_tables=parsed_tables,
            )
            corrected_classes.append(sc_corrected)
            results.append(sc_result)

    fund_corrected["share_classes"] = corrected_classes
    return fund_corrected, results
