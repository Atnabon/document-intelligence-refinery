"""PageIndex schema — hierarchical section tree for deterministic navigation.

The PageIndex provides a table-of-contents–like tree structure over any document,
enabling section-specific queries *without* relying on vector similarity search.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PageNode(BaseModel):
    """A node in the PageIndex tree — represents one section/subsection."""

    node_id: str = Field(
        ..., description="Unique node identifier."
    )
    title: str = Field(
        ..., description="Section title / heading text."
    )
    level: int = Field(
        ...,
        ge=0,
        description="Depth level in the hierarchy (0 = root, 1 = top‑level section, etc.).",
    )
    page_start: int = Field(
        ..., ge=1, description="First page of this section."
    )
    page_end: int = Field(
        ..., ge=1, description="Last page of this section."
    )
    summary: Optional[str] = Field(
        default=None,
        description="LLM-generated summary of this section's content.",
    )
    children: List[PageNode] = Field(
        default_factory=list,
        description="Child sections.",
    )
    ldu_ids: List[str] = Field(
        default_factory=list,
        description="LDU IDs that belong to this section.",
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary metadata (e.g., section_type, importance).",
    )


# Forward reference resolution for recursive model
PageNode.model_rebuild()


class PageIndex(BaseModel):
    """The full hierarchical index tree for a document.

    The tree mirrors the document's logical structure (chapters → sections →
    subsections) and provides deterministic, non-probabilistic navigation.
    """

    document_id: str = Field(
        ..., description="Document this index belongs to."
    )
    root: PageNode = Field(
        ..., description="Root node of the section tree."
    )
    total_sections: int = Field(
        default=0,
        ge=0,
        description="Total number of sections in the tree.",
    )
    max_depth: int = Field(
        default=0,
        ge=0,
        description="Maximum depth of the tree.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "cbe_annual_2023_24",
                "root": {
                    "node_id": "root",
                    "title": "CBE Annual Report 2023-24",
                    "level": 0,
                    "page_start": 1,
                    "page_end": 120,
                    "summary": "Comprehensive annual report of CBE for fiscal year 2023-24.",
                    "children": [
                        {
                            "node_id": "s1",
                            "title": "Financial Highlights",
                            "level": 1,
                            "page_start": 5,
                            "page_end": 15,
                            "summary": "Key financial metrics and KPIs.",
                            "children": [],
                            "ldu_ids": ["cbe_p5_001", "cbe_p6_001"],
                            "metadata": {"section_type": "financial"},
                        }
                    ],
                    "ldu_ids": [],
                    "metadata": {},
                },
                "total_sections": 12,
                "max_depth": 3,
            }
        }
