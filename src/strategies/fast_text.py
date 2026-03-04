"""Strategy A: Fast Text Extractor.

Fastest and cheapest strategy. Uses PyMuPDF to extract text directly from
native-digital PDFs. No layout analysis, no table detection — pure text
extraction with reading-order preservation.

Best for: Simple, single-column, text-dominant native-digital PDFs.
Cost: ~$0.0001 per page (local processing only).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from src.models.document_profile import DocumentProfile
from src.models.ldu import LDU, LDUType
from src.models.provenance import BoundingBox
from src.strategies.base import BaseExtractor

logger = logging.getLogger(__name__)


class FastTextExtractor(BaseExtractor):
    """Strategy A: Direct text extraction via PyMuPDF.

    Pros:
    - Extremely fast (100+ pages/sec)
    - No external API calls
    - Zero cost

    Cons:
    - Cannot handle tables (extracts as broken text)
    - No layout understanding
    - Useless for scanned documents
    """

    def name(self) -> str:
        return "fast_text"

    def cost_per_page(self) -> float:
        return 0.0001

    def extract(
        self,
        profile: DocumentProfile,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
    ) -> List[LDU]:
        """Extract text blocks from each page as LDUs.

        Each text block detected by PyMuPDF becomes one LDU. Blocks are
        classified as headings, paragraphs, or other based on heuristics.
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        ldus: List[LDU] = []

        page_indices = (
            [p - 1 for p in pages] if pages else range(len(doc))
        )

        for page_idx in page_indices:
            if page_idx < 0 or page_idx >= len(doc):
                continue

            page = doc[page_idx]
            page_number = page_idx + 1
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            seq = 0
            for block in blocks.get("blocks", []):
                if block.get("type") != 0:  # skip image blocks
                    continue

                # Concatenate all lines in this block
                text_lines = []
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    text_lines.append(line_text)

                text = "\n".join(text_lines).strip()
                if not text:
                    continue

                # Classify block type
                ldu_type = self._classify_block(text, block)

                # Build bounding box
                bbox_data = block.get("bbox", None)
                bbox = None
                if bbox_data and len(bbox_data) == 4:
                    bbox = BoundingBox(
                        x0=bbox_data[0],
                        y0=bbox_data[1],
                        x1=bbox_data[2],
                        y1=bbox_data[3],
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                    )

                content_hash = hashlib.sha256(text.encode()).hexdigest()

                ldu = LDU(
                    ldu_id=f"{profile.document_id}_p{page_number}_{seq:03d}",
                    document_id=profile.document_id,
                    ldu_type=ldu_type,
                    content=text,
                    structured_content=None,
                    page_number=page_number,
                    bbox=bbox,
                    content_hash=content_hash,
                    section_heading=None,  # populated later by indexer
                    extraction_strategy=self.name(),
                    confidence=0.85,  # base confidence for direct text extraction
                    sequence_index=seq,
                )
                ldus.append(ldu)
                seq += 1

        doc.close()
        logger.info(
            "FastTextExtractor: Extracted %d LDUs from %s",
            len(ldus),
            pdf_path.name,
        )
        return ldus

    @staticmethod
    def _classify_block(text: str, block: dict) -> LDUType:
        """Classify a text block as heading, paragraph, footnote, etc."""
        # Heading: short text, larger font size
        lines = block.get("lines", [])
        if lines:
            spans = lines[0].get("spans", [])
            if spans:
                font_size = spans[0].get("size", 12)
                is_bold = "bold" in spans[0].get("font", "").lower()

                # Heading heuristics
                if (font_size > 14 or is_bold) and len(text) < 200:
                    return LDUType.HEADING

        # Footnote heuristics
        if re.match(r"^\d+[\.\)]\s", text) and len(text) < 300:
            return LDUType.FOOTNOTE

        # List heuristics
        if re.match(r"^[\-\•\*\▪]\s", text) or re.match(r"^\d+\.\s", text):
            return LDUType.LIST

        # Default to paragraph
        return LDUType.PARAGRAPH
