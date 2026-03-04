"""ExtractedDocument schema — the unified output of the extraction pipeline.

Aggregates the DocumentProfile, all LDUs, the PageIndex, and extraction
metadata into a single container for downstream consumption.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.document_profile import DocumentProfile
from src.models.ldu import LDU
from src.models.page_index import PageIndex


class ExtractionMetrics(BaseModel):
    """Operational metrics for the extraction run."""

    extraction_time_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock extraction time."
    )
    strategy_used: str = Field(
        ..., description="Primary strategy that produced the extraction."
    )
    escalation_count: int = Field(
        default=0,
        ge=0,
        description="How many times the pipeline escalated to a higher-cost strategy.",
    )
    total_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total cost (API calls, VLM tokens, etc.).",
    )
    average_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean confidence across all LDUs.",
    )
    low_confidence_count: int = Field(
        default=0,
        ge=0,
        description="Number of LDUs below the confidence threshold.",
    )


class LedgerEntry(BaseModel):
    """A single entry in the extraction_ledger.jsonl — one per document."""

    document_id: str = Field(..., description="Document identifier.")
    filename: str = Field(..., description="Original filename.")
    strategy_selected: str = Field(
        ..., description="Primary extraction strategy used."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall extraction confidence.",
    )
    cost_estimate_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost for this extraction.",
    )
    ldu_count: int = Field(
        default=0, ge=0, description="Number of LDUs extracted."
    )
    table_count: int = Field(
        default=0, ge=0, description="Number of tables extracted."
    )
    escalated: bool = Field(
        default=False,
        description="Whether the pipeline escalated from the initial strategy.",
    )
    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of extraction.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors or warnings during extraction.",
    )
    needs_human_review: bool = Field(
        default=False,
        description="Flagged True when all strategies returned confidence below threshold.",
    )


class ExtractedDocument(BaseModel):
    """The complete extraction result for a single document.

    This is the final artefact produced by the pipeline, containing everything
    needed for downstream RAG, querying, and audit.
    """

    profile: DocumentProfile = Field(
        ..., description="Triage profile of the document."
    )
    ldus: List[LDU] = Field(
        default_factory=list,
        description="All extracted Logical Document Units.",
    )
    page_index: Optional[PageIndex] = Field(
        default=None,
        description="Hierarchical section tree (populated by indexer agent).",
    )
    metrics: ExtractionMetrics = Field(
        ..., description="Operational metrics for this extraction."
    )
    ledger_entry: LedgerEntry = Field(
        ..., description="Ledger entry for audit trail."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Full raw text (for debugging; omitted in production).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (PDF info dict, etc.).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "profile": {"document_id": "cbe_annual_2023_24", "...": "..."},
                "ldus": [{"ldu_id": "cbe_p42_001", "...": "..."}],
                "page_index": None,
                "metrics": {
                    "extraction_time_seconds": 12.5,
                    "strategy_used": "layout_aware",
                    "escalation_count": 0,
                    "total_cost_usd": 0.12,
                    "average_confidence": 0.89,
                    "low_confidence_count": 3,
                },
                "ledger_entry": {
                    "document_id": "cbe_annual_2023_24",
                    "filename": "CBE ANNUAL REPORT 2023-24.pdf",
                    "strategy_selected": "layout_aware",
                    "confidence_score": 0.89,
                    "cost_estimate_usd": 0.12,
                    "ldu_count": 245,
                    "table_count": 18,
                    "escalated": False,
                    "processed_at": "2026-03-04T12:00:00Z",
                    "errors": [],
                },
                "raw_text": None,
                "metadata": {"title": "CBE Annual Report 2023-24"},
            }
        }
