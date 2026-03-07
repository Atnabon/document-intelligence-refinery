"""Logical Document Unit (LDU) schema — the atomic chunk for RAG.

An LDU is the smallest semantically coherent unit extracted from a document.
Unlike naive fixed‑size chunks, LDUs respect logical boundaries: a table is one
LDU, a captioned figure is one LDU, a section paragraph is one LDU.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.provenance import BoundingBox


class LDUType(str, Enum):
    """Classification of the logical unit type."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    HEADING = "heading"
    LIST = "list"
    FOOTNOTE = "footnote"
    CAPTION = "caption"
    KEY_VALUE = "key_value"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    OTHER = "other"


class LDU(BaseModel):
    """A Logical Document Unit — the atomic extraction element.

    Every piece of content extracted from a document is wrapped in an LDU,
    preserving its type, spatial location, and relationship to other LDUs.
    """

    # ── Identity ──────────────────────────────────────────────────────
    ldu_id: str = Field(
        ..., description="Unique ID for this LDU (e.g., doc_id + page + seq)."
    )
    document_id: str = Field(
        ..., description="Parent document identifier."
    )

    # ── Content ───────────────────────────────────────────────────────
    ldu_type: LDUType = Field(
        ..., description="Semantic type of this unit."
    )
    content: str = Field(
        ..., description="Extracted text content of this unit."
    )
    structured_content: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Structured representation (e.g., JSON table with headers/rows). "
            "Present only for tables, key‑value pairs, etc."
        ),
    )

    # ── Spatial Provenance ────────────────────────────────────────────
    page_number: int = Field(
        ..., ge=1, description="1-based page number where this LDU appears."
    )
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box on the page (if available).",
    )
    content_hash: str = Field(
        ..., description="SHA-256 hash of the content for deduplication."
    )

    # ── Context Links ─────────────────────────────────────────────────
    section_heading: Optional[str] = Field(
        default=None,
        description="Nearest ancestor section heading.",
    )
    parent_ldu_id: Optional[str] = Field(
        default=None,
        description="ID of the parent LDU (e.g., caption → figure).",
    )
    child_ldu_ids: List[str] = Field(
        default_factory=list,
        description="IDs of child LDUs.",
    )

    # ── Cross-references ─────────────────────────────────────────────
    cross_references: List[str] = Field(
        default_factory=list,
        description=(
            "Resolved cross-reference targets (e.g., ['table_ldu_003', 'figure_ldu_007']). "
            "Populated by the ChunkingEngine when patterns like 'see Table 3' are detected."
        ),
    )

    # ── Extraction Metadata ───────────────────────────────────────────
    extraction_strategy: str = Field(
        ..., description="Which strategy produced this LDU."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score.",
    )
    sequence_index: int = Field(
        default=0,
        ge=0,
        description="Reading-order index within the page.",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Approximate token count (word-level) for this LDU.",
    )

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "ldu_id": "cbe_annual_2023_24_p42_003",
                "document_id": "cbe_annual_2023_24",
                "ldu_type": "table",
                "content": "Income Statement for FY 2023-24...",
                "structured_content": {
                    "headers": ["Item", "2023-24", "2022-23"],
                    "rows": [
                        ["Interest Income", "189,234", "156,789"],
                        ["Total Revenue", "245,678", "201,345"],
                    ],
                },
                "page_number": 42,
                "bbox": {
                    "x0": 72.0,
                    "y0": 300.0,
                    "x1": 540.0,
                    "y1": 520.0,
                },
                "content_hash": "sha256_abc123...",
                "section_heading": "Financial Statements",
                "parent_ldu_id": None,
                "child_ldu_ids": [],
                "cross_references": [],
                "extraction_strategy": "layout_aware",
                "confidence": 0.92,
                "sequence_index": 3,
                "token_count": 45,
            }
        }
