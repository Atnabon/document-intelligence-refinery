"""DocumentProfile schema — output of the Triage Agent (Phase 1).

The DocumentProfile captures everything known about a document *before* extraction
begins: its origin type, layout complexity, domain classification, and the
recommended extraction strategy.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OriginType(str, Enum):
    """How the PDF was created — determines which extraction path is viable."""

    NATIVE_DIGITAL = "native_digital"  # Born‑digital; has a character stream
    SCANNED_IMAGE = "scanned_image"  # Image‑only; requires OCR or VLM
    MIXED = "mixed"  # Some pages digital, some scanned


class LayoutComplexity(str, Enum):
    """Structural complexity of the page layouts."""

    SIMPLE = "simple"  # Single‑column, minimal tables
    MODERATE = "moderate"  # Multi‑column OR some tables/figures
    COMPLEX = "complex"  # Multi‑column + heavy tables + footnotes


class DomainHint(str, Enum):
    """Coarse domain classification used for strategy selection."""

    FINANCIAL_REPORT = "financial_report"  # Class A
    LEGAL_AUDIT = "legal_audit"  # Class B (scanned gov/legal)
    TECHNICAL_ASSESSMENT = "technical_assessment"  # Class C
    STRUCTURED_DATA = "structured_data"  # Class D (table‑heavy fiscal)
    UNKNOWN = "unknown"


class ExtractionStrategy(str, Enum):
    """Which extraction strategy tier to use."""

    STRATEGY_A = "fast_text"  # PyMuPDF / pdfplumber direct text
    STRATEGY_B = "layout_aware"  # Layout‑preserving with table detection
    STRATEGY_C = "vision_model"  # VLM‑based (GPT‑4o / Gemini vision)


class DocumentProfile(BaseModel):
    """Profile produced by the Triage Agent for every ingested document.

    This is the *routing descriptor* that the ExtractionRouter consumes to
    select the correct extraction strategy.
    """

    # ── Identity ──────────────────────────────────────────────────────
    document_id: str = Field(
        ..., description="Unique identifier (typically filename stem or UUID)."
    )
    filename: str = Field(..., description="Original filename.")
    file_hash: str = Field(
        ..., description="SHA-256 hash of the source file for deduplication."
    )

    # ── Classification ────────────────────────────────────────────────
    origin_type: OriginType = Field(
        ..., description="Whether the PDF is native digital, scanned, or mixed."
    )
    layout_complexity: LayoutComplexity = Field(
        ..., description="Structural complexity of the document layout."
    )
    domain_hint: DomainHint = Field(
        default=DomainHint.UNKNOWN,
        description="Best‑guess domain classification.",
    )

    # ── Metadata ──────────────────────────────────────────────────────
    page_count: int = Field(..., ge=1, description="Total number of pages.")
    language: str = Field(
        default="en", description="Detected primary language (ISO 639-1)."
    )
    has_tables: bool = Field(
        default=False, description="Whether tables were detected during triage."
    )
    has_images: bool = Field(
        default=False,
        description="Whether embedded images/figures were detected.",
    )
    has_footnotes: bool = Field(
        default=False, description="Whether footnotes/endnotes were detected."
    )
    scanned_page_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of pages that appear to be scanned images.",
    )

    # ── Strategy Decision ─────────────────────────────────────────────
    recommended_strategy: ExtractionStrategy = Field(
        ..., description="The extraction strategy selected by the Triage Agent."
    )
    strategy_rationale: str = Field(
        default="",
        description="Human‑readable explanation for why this strategy was chosen.",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated extraction cost in USD.",
    )

    # ── Provenance ────────────────────────────────────────────────────
    profiled_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of profiling.",
    )

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "document_id": "cbe_annual_2023_24",
                "filename": "CBE ANNUAL REPORT 2023-24.pdf",
                "file_hash": "a1b2c3d4e5f6...",
                "origin_type": "native_digital",
                "layout_complexity": "complex",
                "domain_hint": "financial_report",
                "page_count": 120,
                "language": "en",
                "has_tables": True,
                "has_images": True,
                "has_footnotes": True,
                "scanned_page_ratio": 0.0,
                "recommended_strategy": "layout_aware",
                "strategy_rationale": "Native digital with complex multi-column layout and heavy tables. Layout-aware extraction preferred for table fidelity.",
                "estimated_cost_usd": 0.12,
                "profiled_at": "2026-03-04T12:00:00Z",
            }
        }
