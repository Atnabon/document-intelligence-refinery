"""Confidence score calculators for extraction strategies.

Each strategy has a tailored confidence model based on signals available
at extraction time. The `should_escalate` function implements the
confidence-gated escalation decision.
"""

from __future__ import annotations

from typing import Dict, Any


def calculate_text_confidence(
    text_coverage: float,
    table_quality: float = 0.0,
    has_structure: bool = False,
) -> float:
    """Calculate confidence for Strategy A (fast text extraction).

    Signals:
        - text_coverage (40%): fraction of page area with text
        - table_quality (30%): quality of detected tables (0 if none)
        - has_structure (30%): whether structural elements were found

    Returns:
        Float in [0, 1].
    """
    score = 0.0
    score += min(text_coverage, 1.0) * 0.40
    score += min(table_quality, 1.0) * 0.30
    score += (0.30 if has_structure else 0.0)
    return round(min(score, 1.0), 3)


def calculate_layout_confidence(
    table_quality: float,
    block_structure: float,
    coverage: float,
) -> float:
    """Calculate confidence for Strategy B (layout-aware extraction).

    Signals:
        - table_quality (50%): quality of table extraction
        - block_structure (30%): quality of text block detection
        - coverage (20%): text coverage fraction

    Returns:
        Float in [0, 1].
    """
    score = 0.0
    score += min(table_quality, 1.0) * 0.50
    score += min(block_structure, 1.0) * 0.30
    score += min(coverage, 1.0) * 0.20
    return round(min(score, 1.0), 3)


def calculate_vision_confidence(
    base_confidence: float = 0.90,
    coverage: float = 1.0,
    has_complex_layout: bool = False,
) -> float:
    """Calculate confidence for Strategy C (vision/OCR extraction).

    VLM/OCR is generally high confidence but reduced for low coverage.

    Returns:
        Float in [0, 1].
    """
    score = base_confidence
    if coverage < 0.5:
        score -= 0.15
    if has_complex_layout:
        score -= 0.05
    return round(max(0.0, min(score, 1.0)), 3)


def should_escalate(
    current_confidence: float,
    threshold: float = 0.70,
    current_cost: float = 0.0,
    budget_remaining: float = float("inf"),
    next_strategy_cost: float = 0.0,
) -> bool:
    """Decide whether to escalate to the next strategy tier.

    Escalates if:
    1. Current confidence is below the threshold, AND
    2. Budget allows the next strategy.

    Args:
        current_confidence: Confidence from current strategy.
        threshold: Minimum acceptable confidence.
        current_cost: Cost spent so far.
        budget_remaining: Remaining budget in USD.
        next_strategy_cost: Cost of the next strategy.

    Returns:
        True if escalation is recommended.
    """
    if current_confidence >= threshold:
        return False
    if next_strategy_cost > budget_remaining:
        return False
    return True
