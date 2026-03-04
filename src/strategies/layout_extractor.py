"""Strategy B: Layout-Aware Extractor.

Mid-tier strategy that preserves document structure. Uses pdfplumber for
table detection and PyMuPDF for text, combining both to produce layout-faithful
LDUs with structured table output.

Best for: Native-digital PDFs with complex layouts, multi-column text, and tables.
Cost: ~$0.001 per page (local processing, higher CPU usage).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import pdfplumber

from src.models.document_profile import DocumentProfile
from src.models.ldu import LDU, LDUType
from src.models.provenance import BoundingBox
from src.strategies.base import BaseExtractor

logger = logging.getLogger(__name__)


class LayoutExtractor(BaseExtractor):
    """Strategy B: Layout-aware extraction with table detection.

    Pros:
    - Tables extracted as structured JSON (headers + rows)
    - Respects multi-column layouts
    - No external API calls

    Cons:
    - Slower than Strategy A (~10 pages/sec)
    - Cannot handle scanned documents
    - Table detection can fail on unusual layouts
    """

    # Minimum confidence to accept a detected table
    TABLE_CONFIDENCE_THRESHOLD: float = 0.6

    def name(self) -> str:
        return "layout_aware"

    def cost_per_page(self) -> float:
        return 0.001

    def extract(
        self,
        profile: DocumentProfile,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
    ) -> List[LDU]:
        """Extract LDUs with layout awareness and table structure.

        Two-pass extraction:
        1. pdfplumber pass: detect and extract tables as structured JSON
        2. PyMuPDF pass: extract remaining text blocks, excluding table regions
        """
        pdf_path = Path(pdf_path)
        ldus: List[LDU] = []

        # ── Pass 1: Table extraction via pdfplumber ───────────────────
        table_ldus, table_regions = self._extract_tables(profile, pdf_path, pages)
        ldus.extend(table_ldus)

        # ── Pass 2: Text extraction via PyMuPDF (excluding table regions)
        text_ldus = self._extract_text_blocks(
            profile, pdf_path, pages, table_regions
        )
        ldus.extend(text_ldus)

        # Sort by page number and sequence
        ldus.sort(key=lambda l: (l.page_number, l.sequence_index))

        logger.info(
            "LayoutExtractor: Extracted %d LDUs (%d tables) from %s",
            len(ldus),
            len(table_ldus),
            pdf_path.name,
        )
        return ldus

    def _extract_tables(
        self,
        profile: DocumentProfile,
        pdf_path: Path,
        pages: Optional[list[int]],
    ) -> tuple[List[LDU], Dict[int, List[tuple]]]:
        """Extract tables using pdfplumber.

        Returns:
            (table_ldus, table_regions) — LDUs and bbox regions to exclude from text pass.
        """
        table_ldus: List[LDU] = []
        table_regions: Dict[int, List[tuple]] = {}  # page_num -> list of bboxes

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_indices = (
                    [p - 1 for p in pages] if pages else range(len(pdf.pages))
                )

                for page_idx in page_indices:
                    if page_idx < 0 or page_idx >= len(pdf.pages):
                        continue

                    page = pdf.pages[page_idx]
                    page_number = page_idx + 1
                    tables = page.find_tables()

                    for t_idx, table in enumerate(tables):
                        try:
                            rows = table.extract()
                            if not rows or len(rows) < 2:
                                continue

                            # First row as headers, rest as data
                            headers = [
                                str(h).strip() if h else f"col_{i}"
                                for i, h in enumerate(rows[0])
                            ]
                            data_rows = [
                                [str(cell).strip() if cell else "" for cell in row]
                                for row in rows[1:]
                            ]

                            structured = {
                                "headers": headers,
                                "rows": data_rows,
                                "row_count": len(data_rows),
                                "col_count": len(headers),
                            }

                            # Build text representation
                            text = self._table_to_text(headers, data_rows)
                            content_hash = hashlib.sha256(text.encode()).hexdigest()

                            # Get table bbox
                            bbox_data = table.bbox
                            bbox = None
                            if bbox_data:
                                bbox = BoundingBox(
                                    x0=bbox_data[0],
                                    y0=bbox_data[1],
                                    x1=bbox_data[2],
                                    y1=bbox_data[3],
                                    page_width=float(page.width),
                                    page_height=float(page.height),
                                )
                                # Track table region for exclusion
                                table_regions.setdefault(page_number, []).append(
                                    bbox_data
                                )

                            ldu = LDU(
                                ldu_id=f"{profile.document_id}_p{page_number}_t{t_idx:02d}",
                                document_id=profile.document_id,
                                ldu_type=LDUType.TABLE,
                                content=text,
                                structured_content=structured,
                                page_number=page_number,
                                bbox=bbox,
                                content_hash=content_hash,
                                extraction_strategy=self.name(),
                                confidence=0.90,
                                sequence_index=t_idx * 100,  # tables get high seq
                            )
                            table_ldus.append(ldu)

                        except Exception as e:
                            logger.warning(
                                "Table extraction failed on page %d, table %d: %s",
                                page_number,
                                t_idx,
                                str(e),
                            )

        except Exception as e:
            logger.error("pdfplumber failed on %s: %s", pdf_path.name, str(e))

        return table_ldus, table_regions

    def _extract_text_blocks(
        self,
        profile: DocumentProfile,
        pdf_path: Path,
        pages: Optional[list[int]],
        table_regions: Dict[int, List[tuple]],
    ) -> List[LDU]:
        """Extract non-table text blocks via PyMuPDF."""
        text_ldus: List[LDU] = []
        doc = fitz.open(str(pdf_path))

        page_indices = [p - 1 for p in pages] if pages else range(len(doc))

        for page_idx in page_indices:
            if page_idx < 0 or page_idx >= len(doc):
                continue

            page = doc[page_idx]
            page_number = page_idx + 1
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            regions = table_regions.get(page_number, [])
            seq = 0

            for block in blocks.get("blocks", []):
                if block.get("type") != 0:
                    continue

                bbox_data = block.get("bbox", (0, 0, 0, 0))

                # Skip if this block overlaps a table region
                if self._overlaps_any(bbox_data, regions):
                    continue

                text_lines = []
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    text_lines.append(line_text)

                text = "\n".join(text_lines).strip()
                if not text:
                    continue

                ldu_type = self._classify_block(text, block)
                content_hash = hashlib.sha256(text.encode()).hexdigest()

                bbox = BoundingBox(
                    x0=bbox_data[0],
                    y0=bbox_data[1],
                    x1=bbox_data[2],
                    y1=bbox_data[3],
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                )

                ldu = LDU(
                    ldu_id=f"{profile.document_id}_p{page_number}_{seq:03d}",
                    document_id=profile.document_id,
                    ldu_type=ldu_type,
                    content=text,
                    structured_content=None,
                    page_number=page_number,
                    bbox=bbox,
                    content_hash=content_hash,
                    extraction_strategy=self.name(),
                    confidence=0.88,
                    sequence_index=seq,
                )
                text_ldus.append(ldu)
                seq += 1

        doc.close()
        return text_ldus

    @staticmethod
    def _overlaps_any(
        block_bbox: tuple, table_regions: List[tuple], threshold: float = 0.3
    ) -> bool:
        """Check if a text block significantly overlaps any table region."""
        bx0, by0, bx1, by1 = block_bbox
        for tx0, ty0, tx1, ty1 in table_regions:
            # Compute intersection
            ix0 = max(bx0, tx0)
            iy0 = max(by0, ty0)
            ix1 = min(bx1, tx1)
            iy1 = min(by1, ty1)

            if ix0 < ix1 and iy0 < iy1:
                intersection = (ix1 - ix0) * (iy1 - iy0)
                block_area = max((bx1 - bx0) * (by1 - by0), 1)
                if intersection / block_area > threshold:
                    return True
        return False

    @staticmethod
    def _table_to_text(headers: List[str], rows: List[List[str]]) -> str:
        """Convert table to readable text representation."""
        lines = [" | ".join(headers)]
        lines.append(" | ".join(["---"] * len(headers)))
        for row in rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)

    @staticmethod
    def _classify_block(text: str, block: dict) -> LDUType:
        """Classify a text block type using heuristics."""
        lines = block.get("lines", [])
        if lines:
            spans = lines[0].get("spans", [])
            if spans:
                font_size = spans[0].get("size", 12)
                is_bold = "bold" in spans[0].get("font", "").lower()
                if (font_size > 14 or is_bold) and len(text) < 200:
                    return LDUType.HEADING

        if re.match(r"^\d+[\.\)]\s", text) and len(text) < 300:
            return LDUType.FOOTNOTE

        if re.match(r"^[\-\•\*\▪]\s", text) or re.match(r"^\d+\.\s", text):
            return LDUType.LIST

        return LDUType.PARAGRAPH
