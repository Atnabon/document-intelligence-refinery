"""ProvenanceChain schema — spatial provenance for every extracted claim.

Provenance is the critical trust layer: every fact extracted from a document must
carry a chain of evidence linking it back to the *exact* location in the source
PDF where it appears.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates on a page (PDF coordinate system, origin at bottom-left)."""

    x0: float = Field(..., description="Left edge x-coordinate.")
    y0: float = Field(..., description="Bottom edge y-coordinate.")
    x1: float = Field(..., description="Right edge x-coordinate.")
    y1: float = Field(..., description="Top edge y-coordinate.")
    page_width: float = Field(default=612.0, description="Page width in points.")
    page_height: float = Field(default=792.0, description="Page height in points.")


class SourceCitation(BaseModel):
    """A single source location within a document."""

    page_number: int = Field(..., ge=1, description="1-based page number.")
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box of the cited region. None if not available.",
    )
    text_snippet: str = Field(
        ...,
        max_length=500,
        description="The exact text snippet from the source at this location.",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash of the text_snippet for integrity verification.",
    )
    section_heading: Optional[str] = Field(
        default=None,
        description="Nearest section heading above this citation.",
    )


class ProvenanceChain(BaseModel):
    """Full provenance chain for an extracted fact or answer.

    Links a claim/answer back to one or more precise locations in the source
    document, enabling audit and verification.
    """

    document_id: str = Field(
        ..., description="ID of the source document."
    )
    document_name: str = Field(
        ..., description="Human-readable filename of the source document."
    )
    claim: str = Field(
        ..., description="The extracted fact or answer being cited."
    )
    citations: List[SourceCitation] = Field(
        ...,
        min_length=1,
        description="One or more source citations supporting this claim.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this provenance chain.",
    )
    verification_status: str = Field(
        default="unverified",
        description="One of: verified, unverified, unverifiable.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "cbe_annual_2023_24",
                "claim": "Total assets reached ETB 2.1 trillion in FY 2023-24.",
                "citations": [
                    {
                        "page_number": 42,
                        "bbox": {
                            "x0": 72.0,
                            "y0": 500.0,
                            "x1": 540.0,
                            "y1": 520.0,
                            "page_width": 612.0,
                            "page_height": 792.0,
                        },
                        "text_snippet": "Total assets reached ETB 2.1 trillion...",
                        "content_hash": "abc123def456...",
                        "section_heading": "Financial Highlights",
                    }
                ],
                "confidence": 0.95,
                "verification_status": "verified",
            }
        }
