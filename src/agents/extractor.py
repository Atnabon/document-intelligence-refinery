"""Extraction Router Agent — Confidence-Gated Strategy Escalation.

The ExtractionRouter is the orchestrator of the extraction pipeline. It:
1. Receives a DocumentProfile from the Triage Agent
2. Selects the recommended extraction strategy
3. Runs extraction and evaluates confidence
4. Escalates to a higher-tier strategy if confidence is below threshold
5. Produces the final ExtractedDocument with ledger entry
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.models.document_profile import DocumentProfile, ExtractionStrategy
from src.models.extracted_document import (
    ExtractedDocument,
    ExtractionMetrics,
    LedgerEntry,
)
from src.models.ldu import LDU
from src.strategies.base import BaseExtractor
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_extractor import LayoutExtractor
from src.strategies.vision_extractor import VisionExtractor

logger = logging.getLogger(__name__)


class ExtractionRouter:
    """Routes documents to the appropriate extraction strategy with
    confidence-gated escalation.

    Escalation Guard:
    If the initial strategy produces LDUs with average confidence below
    the escalation threshold, the router automatically escalates to the
    next tier strategy:
        Strategy A → Strategy B → Strategy C

    Cost Budget Guard:
    Each document has a maximum extraction cost budget. If escalating
    would exceed the budget, the router returns the best available result
    with a warning.
    """

    # ── Configuration ─────────────────────────────────────────────────
    CONFIDENCE_ESCALATION_THRESHOLD: float = 0.70
    MAX_COST_PER_DOCUMENT_USD: float = 5.0
    LOW_CONFIDENCE_THRESHOLD: float = 0.60

    # Strategy escalation order
    ESCALATION_CHAIN = [
        ExtractionStrategy.STRATEGY_A,
        ExtractionStrategy.STRATEGY_B,
        ExtractionStrategy.STRATEGY_C,
    ]

    def __init__(self, config: Optional[dict] = None):
        """Initialize with strategy instances and optional config."""
        self.strategies: Dict[str, BaseExtractor] = {
            "fast_text": FastTextExtractor(),
            "layout_aware": LayoutExtractor(),
            "vision_model": VisionExtractor(),
        }

        if config:
            self.CONFIDENCE_ESCALATION_THRESHOLD = config.get(
                "confidence_escalation_threshold",
                self.CONFIDENCE_ESCALATION_THRESHOLD,
            )
            self.MAX_COST_PER_DOCUMENT_USD = config.get(
                "max_cost_per_document_usd",
                self.MAX_COST_PER_DOCUMENT_USD,
            )

    def extract_document(
        self,
        profile: DocumentProfile,
        pdf_path: str | Path,
    ) -> ExtractedDocument:
        """Run extraction with confidence-gated escalation.

        Args:
            profile: DocumentProfile from triage.
            pdf_path: Path to the source PDF.

        Returns:
            Fully populated ExtractedDocument.
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()

        strategy_name = profile.recommended_strategy
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        logger.info(
            "ExtractionRouter: Starting with strategy '%s' for %s",
            strategy_name,
            profile.filename,
        )

        # ── Initial Extraction ────────────────────────────────────────
        ldus = strategy.extract(profile, pdf_path)
        avg_confidence = self._compute_avg_confidence(ldus)
        total_cost = strategy.cost_per_page() * profile.page_count
        escalation_count = 0
        errors: List[str] = []

        # ── Escalation Guard ──────────────────────────────────────────
        current_strategy_idx = self._strategy_index(strategy_name)
        while (
            avg_confidence < self.CONFIDENCE_ESCALATION_THRESHOLD
            and current_strategy_idx < len(self.ESCALATION_CHAIN) - 1
        ):
            next_strategy_enum = self.ESCALATION_CHAIN[current_strategy_idx + 1]
            next_strategy_name = next_strategy_enum.value
            next_strategy = self.strategies.get(next_strategy_name)

            if not next_strategy:
                break

            # Cost budget check
            escalation_cost = next_strategy.cost_per_page() * profile.page_count
            if total_cost + escalation_cost > self.MAX_COST_PER_DOCUMENT_USD:
                warning = (
                    f"Escalation to '{next_strategy_name}' would exceed cost budget "
                    f"(${total_cost + escalation_cost:.4f} > ${self.MAX_COST_PER_DOCUMENT_USD:.2f}). "
                    f"Keeping current result with confidence {avg_confidence:.3f}."
                )
                logger.warning(warning)
                errors.append(warning)
                break

            logger.info(
                "Escalating from '%s' (confidence=%.3f) to '%s'",
                strategy_name,
                avg_confidence,
                next_strategy_name,
            )

            # Run escalated strategy
            try:
                escalated_ldus = next_strategy.extract(profile, pdf_path)
                escalated_confidence = self._compute_avg_confidence(escalated_ldus)

                if escalated_confidence > avg_confidence:
                    ldus = escalated_ldus
                    avg_confidence = escalated_confidence
                    strategy_name = next_strategy_name
                    total_cost += escalation_cost
                    escalation_count += 1
                else:
                    logger.info(
                        "Escalation did not improve confidence (%.3f vs %.3f). Keeping original.",
                        escalated_confidence,
                        avg_confidence,
                    )
                    break

            except Exception as e:
                error_msg = f"Escalation to '{next_strategy_name}' failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                break

            current_strategy_idx += 1

        # ── Build Result ──────────────────────────────────────────────
        elapsed = time.time() - start_time
        low_conf_count = sum(
            1 for ldu in ldus if ldu.confidence < self.LOW_CONFIDENCE_THRESHOLD
        )
        table_count = sum(1 for ldu in ldus if ldu.ldu_type == "table")

        metrics = ExtractionMetrics(
            extraction_time_seconds=round(elapsed, 2),
            strategy_used=strategy_name,
            escalation_count=escalation_count,
            total_cost_usd=round(total_cost, 6),
            average_confidence=round(avg_confidence, 4),
            low_confidence_count=low_conf_count,
        )

        ledger = LedgerEntry(
            document_id=profile.document_id,
            filename=profile.filename,
            strategy_selected=strategy_name,
            confidence_score=round(avg_confidence, 4),
            cost_estimate_usd=round(total_cost, 6),
            ldu_count=len(ldus),
            table_count=table_count,
            escalated=escalation_count > 0,
            processed_at=datetime.utcnow(),
            errors=errors,
        )

        return ExtractedDocument(
            profile=profile,
            ldus=ldus,
            page_index=None,  # populated by indexer agent (Phase 3)
            metrics=metrics,
            ledger_entry=ledger,
        )

    def append_to_ledger(
        self, ledger_entry: LedgerEntry, ledger_path: str | Path
    ) -> None:
        """Append a ledger entry to the extraction_ledger.jsonl file."""
        ledger_path = Path(ledger_path)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        with open(ledger_path, "a") as f:
            f.write(ledger_entry.model_dump_json() + "\n")

        logger.info("Ledger entry written for %s", ledger_entry.document_id)

    @staticmethod
    def _compute_avg_confidence(ldus: List[LDU]) -> float:
        """Compute average confidence across LDUs."""
        if not ldus:
            return 0.0
        return sum(ldu.confidence for ldu in ldus) / len(ldus)

    def _strategy_index(self, strategy_name: str) -> int:
        """Get the index of a strategy in the escalation chain."""
        name_to_enum = {
            "fast_text": ExtractionStrategy.STRATEGY_A,
            "layout_aware": ExtractionStrategy.STRATEGY_B,
            "vision_model": ExtractionStrategy.STRATEGY_C,
        }
        enum_val = name_to_enum.get(strategy_name)
        if enum_val and enum_val in self.ESCALATION_CHAIN:
            return self.ESCALATION_CHAIN.index(enum_val)
        return len(self.ESCALATION_CHAIN) - 1
