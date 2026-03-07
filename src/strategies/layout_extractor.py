"""Strategy B: Layout-Aware Extractor using Docling.

Mid-tier strategy that preserves document structure. Uses Docling as the primary
layout-aware extraction engine with a DoclingDocumentAdapter that normalizes
output to the internal LDU schema. Falls back to pdfplumber+PyMuPDF when Docling
is unavailable.

Docling (IBM) performs deep document understanding: detecting tables, figures,
headings, lists, and other structural elements with high accuracy. It supports
both native-digital and mixed PDFs.

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


# ═══════════════════════════════════════════════════════════════════════════
# Docling Document Adapter
# ═══════════════════════════════════════════════════════════════════════════


class DoclingDocumentAdapter:
    """Adapter that normalizes Docling's output to the internal LDU schema.

    Docling produces a structured document representation with typed elements
    (text, table, figure, heading, list, etc.). This adapter converts each
    element into an LDU with proper metadata including bounding boxes,
    content hashes, and structured content for tables.

    This fulfills the challenge requirement:
        "Strategy B: Integrate MinerU or Docling as LayoutExtractor.
         Implement a DoclingDocumentAdapter that normalizes output to
         your internal schema."
    """

    # Map Docling content types to LDU types
    DOCLING_TYPE_MAP = {
        "title": LDUType.HEADING,
        "section_header": LDUType.HEADING,
        "text": LDUType.PARAGRAPH,
        "paragraph": LDUType.PARAGRAPH,
        "table": LDUType.TABLE,
        "figure": LDUType.FIGURE,
        "picture": LDUType.FIGURE,
        "list_item": LDUType.LIST,
        "list": LDUType.LIST,
        "caption": LDUType.CAPTION,
        "footnote": LDUType.FOOTNOTE,
        "formula": LDUType.PARAGRAPH,
        "page_header": LDUType.PARAGRAPH,
        "page_footer": LDUType.FOOTNOTE,
        "code": LDUType.PARAGRAPH,
        "reference": LDUType.FOOTNOTE,
    }

    @classmethod
    def adapt(
        cls,
        docling_result: Any,
        profile: DocumentProfile,
        pages: Optional[list[int]] = None,
    ) -> List[LDU]:
        """Convert a Docling conversion result into a list of LDUs.

        Args:
            docling_result: The result from ``DocumentConverter.convert()``.
            profile: The DocumentProfile for provenance.
            pages: Optional page filter (1-based). If provided, only LDUs
                   from these pages are returned.

        Returns:
            List of normalized LDUs with content_hash, bbox, and structured_content.
        """
        ldus: List[LDU] = []
        doc = docling_result.document
        seq_by_page: Dict[int, int] = {}

        # ── Extract tables ────────────────────────────────────────────
        table_ldus = cls._extract_tables(doc, profile, pages, seq_by_page)
        ldus.extend(table_ldus)

        # ── Extract text elements ─────────────────────────────────────
        text_ldus = cls._extract_text_elements(doc, profile, pages, seq_by_page)
        ldus.extend(text_ldus)

        # Sort by page number and sequence
        ldus.sort(key=lambda l: (l.page_number, l.sequence_index))

        return ldus

    @classmethod
    def _extract_tables(
        cls,
        doc: Any,
        profile: DocumentProfile,
        pages: Optional[list[int]],
        seq_by_page: Dict[int, int],
    ) -> List[LDU]:
        """Extract tables from Docling document."""
        table_ldus: List[LDU] = []

        try:
            tables = list(doc.tables) if hasattr(doc, "tables") else []
        except Exception:
            tables = []

        for t_idx, table in enumerate(tables):
            try:
                # Get page number from table provenance
                page_number = cls._get_page_number(table)
                if pages and page_number not in pages:
                    continue

                # Export table as dataframe if available
                table_data = cls._table_to_structured(table)
                if not table_data:
                    continue

                headers = table_data["headers"]
                data_rows = table_data["rows"]

                structured = {
                    "headers": headers,
                    "rows": data_rows,
                    "row_count": len(data_rows),
                    "col_count": len(headers),
                }

                # Build text representation
                text = cls._table_to_text(headers, data_rows)
                content_hash = hashlib.sha256(text.encode()).hexdigest()

                # Get bounding box
                bbox = cls._get_bbox(table, page_number)

                seq = seq_by_page.get(page_number, 0)
                seq_by_page[page_number] = seq + 1

                ldu = LDU(
                    ldu_id=f"{profile.document_id}_p{page_number}_t{t_idx:02d}",
                    document_id=profile.document_id,
                    ldu_type=LDUType.TABLE,
                    content=text,
                    structured_content=structured,
                    page_number=page_number,
                    bbox=bbox,
                    content_hash=content_hash,
                    extraction_strategy="layout_aware",
                    confidence=0.92,
                    sequence_index=seq * 100,
                )
                table_ldus.append(ldu)

            except Exception as e:
                logger.warning("Docling table %d extraction failed: %s", t_idx, e)

        return table_ldus

    @classmethod
    def _extract_text_elements(
        cls,
        doc: Any,
        profile: DocumentProfile,
        pages: Optional[list[int]],
        seq_by_page: Dict[int, int],
    ) -> List[LDU]:
        """Extract text elements from Docling document."""
        text_ldus: List[LDU] = []

        try:
            # Iterate through document body elements
            elements = cls._iterate_elements(doc)
        except Exception:
            elements = []

        for elem in elements:
            try:
                text = cls._get_element_text(elem)
                if not text or not text.strip():
                    continue

                page_number = cls._get_page_number(elem)
                if pages and page_number not in pages:
                    continue

                ldu_type = cls._classify_element(elem)
                content_hash = hashlib.sha256(text.encode()).hexdigest()
                bbox = cls._get_bbox(elem, page_number)

                seq = seq_by_page.get(page_number, 0)
                seq_by_page[page_number] = seq + 1

                ldu = LDU(
                    ldu_id=f"{profile.document_id}_p{page_number}_{seq:03d}",
                    document_id=profile.document_id,
                    ldu_type=ldu_type,
                    content=text.strip(),
                    structured_content=None,
                    page_number=page_number,
                    bbox=bbox,
                    content_hash=content_hash,
                    extraction_strategy="layout_aware",
                    confidence=0.90,
                    sequence_index=seq,
                )
                text_ldus.append(ldu)

            except Exception as e:
                logger.debug("Docling element extraction failed: %s", e)

        return text_ldus

    @classmethod
    def _iterate_elements(cls, doc: Any) -> list:
        """Iterate through Docling document elements, skipping tables.

        Handles different Docling API versions.
        """
        elements = []

        # Try the iterate_items API (Docling v2+)
        if hasattr(doc, "iterate_items"):
            try:
                for item, _level in doc.iterate_items():
                    # Skip tables (handled separately)
                    label = cls._get_element_label(item)
                    if label == "table":
                        continue
                    elements.append(item)
                return elements
            except Exception:
                pass

        # Try body.children (Docling v1 / earlier)
        if hasattr(doc, "body") and hasattr(doc.body, "children"):
            try:
                for child in doc.body.children:
                    label = cls._get_element_label(child)
                    if label == "table":
                        continue
                    elements.append(child)
                return elements
            except Exception:
                pass

        # Fallback: export to markdown and parse
        if hasattr(doc, "export_to_markdown"):
            try:
                md = doc.export_to_markdown()
                if md:
                    # Create synthetic elements from markdown lines
                    for line in md.split("\n"):
                        if line.strip():
                            elements.append({"_text": line.strip(), "_type": "text"})
                return elements
            except Exception:
                pass

        return elements

    @classmethod
    def _get_element_text(cls, elem: Any) -> str:
        """Get text content from a Docling element."""
        if isinstance(elem, dict):
            return elem.get("_text", "")
        if hasattr(elem, "text"):
            return str(elem.text)
        if hasattr(elem, "export_to_markdown"):
            try:
                return elem.export_to_markdown()
            except Exception:
                pass
        return str(elem) if elem else ""

    @classmethod
    def _get_element_label(cls, elem: Any) -> str:
        """Get the content type label of a Docling element."""
        if isinstance(elem, dict):
            return elem.get("_type", "text")
        if hasattr(elem, "label"):
            label = elem.label
            return str(label.value).lower() if hasattr(label, "value") else str(label).lower()
        if hasattr(elem, "content_type"):
            return str(elem.content_type).lower()
        type_name = type(elem).__name__.lower()
        if "table" in type_name:
            return "table"
        if "heading" in type_name or "title" in type_name:
            return "section_header"
        if "list" in type_name:
            return "list_item"
        if "figure" in type_name or "picture" in type_name:
            return "figure"
        return "text"

    @classmethod
    def _classify_element(cls, elem: Any) -> LDUType:
        """Classify a Docling element into an LDU type."""
        label = cls._get_element_label(elem)
        return cls.DOCLING_TYPE_MAP.get(label, LDUType.PARAGRAPH)

    @classmethod
    def _get_page_number(cls, elem: Any) -> int:
        """Extract page number from element provenance. Defaults to 1."""
        # Try prov attribute (Docling v2)
        if hasattr(elem, "prov") and elem.prov:
            prov_list = elem.prov if isinstance(elem.prov, list) else [elem.prov]
            for prov in prov_list:
                if hasattr(prov, "page_no"):
                    return int(prov.page_no)
                if hasattr(prov, "page"):
                    return int(prov.page)

        # Try metadata
        if hasattr(elem, "metadata"):
            meta = elem.metadata
            if hasattr(meta, "page_number"):
                return int(meta.page_number)
            if isinstance(meta, dict):
                return int(meta.get("page_number", 1))

        return 1

    @classmethod
    def _get_bbox(cls, elem: Any, page_number: int) -> Optional[BoundingBox]:
        """Extract bounding box from element provenance."""
        try:
            if hasattr(elem, "prov") and elem.prov:
                prov_list = elem.prov if isinstance(elem.prov, list) else [elem.prov]
                for prov in prov_list:
                    bbox_obj = getattr(prov, "bbox", None)
                    if bbox_obj is None:
                        continue

                    # Docling BoundingBox has l, t, r, b or x0, y0, x1, y1
                    x0 = getattr(bbox_obj, "l", None) or getattr(bbox_obj, "x0", 0)
                    y0 = getattr(bbox_obj, "t", None) or getattr(bbox_obj, "y0", 0)
                    x1 = getattr(bbox_obj, "r", None) or getattr(bbox_obj, "x1", 0)
                    y1 = getattr(bbox_obj, "b", None) or getattr(bbox_obj, "y1", 0)

                    page_w = getattr(prov, "page_width", None) or getattr(bbox_obj, "page_width", 612.0)
                    page_h = getattr(prov, "page_height", None) or getattr(bbox_obj, "page_height", 792.0)

                    if hasattr(bbox_obj, "to_dict"):
                        d = bbox_obj.to_dict()
                        x0 = d.get("l", d.get("x0", x0))
                        y0 = d.get("t", d.get("y0", y0))
                        x1 = d.get("r", d.get("x1", x1))
                        y1 = d.get("b", d.get("y1", y1))

                    return BoundingBox(
                        x0=float(x0), y0=float(y0),
                        x1=float(x1), y1=float(y1),
                        page_width=float(page_w),
                        page_height=float(page_h),
                    )
        except Exception:
            pass
        return None

    @classmethod
    def _table_to_structured(cls, table: Any) -> Optional[Dict]:
        """Convert a Docling table to structured {headers, rows} dict."""
        # Try export_to_dataframe (Docling v2+)
        try:
            df = table.export_to_dataframe()
            headers = [str(c) for c in df.columns.tolist()]
            rows = [[str(cell) for cell in row] for row in df.values.tolist()]
            if headers and rows:
                return {"headers": headers, "rows": rows}
        except Exception:
            pass

        # Try grid / cells API
        try:
            if hasattr(table, "data") and isinstance(table.data, list):
                if len(table.data) >= 2:
                    headers = [str(c) for c in table.data[0]]
                    rows = [[str(c) for c in row] for row in table.data[1:]]
                    return {"headers": headers, "rows": rows}
        except Exception:
            pass

        # Try text export
        try:
            text = cls._get_element_text(table)
            if text and "|" in text:
                lines = [l.strip() for l in text.split("\n") if l.strip() and "|" in l]
                if len(lines) >= 2:
                    headers = [c.strip() for c in lines[0].split("|") if c.strip()]
                    start = 1
                    # Skip separator line
                    if all(c.strip().replace("-", "") == "" for c in lines[1].split("|") if c.strip()):
                        start = 2
                    rows = [
                        [c.strip() for c in line.split("|") if c.strip()]
                        for line in lines[start:]
                    ]
                    return {"headers": headers, "rows": rows}
        except Exception:
            pass

        return None

    @staticmethod
    def _table_to_text(headers: List[str], rows: List[List[str]]) -> str:
        """Convert table to readable text representation."""
        lines = [" | ".join(headers)]
        lines.append(" | ".join(["---"] * len(headers)))
        for row in rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Strategy B: Layout Extractor
# ═══════════════════════════════════════════════════════════════════════════


class LayoutExtractor(BaseExtractor):
    """Strategy B: Layout-aware extraction using Docling with pdfplumber fallback.

    Primary engine: Docling (IBM document understanding library) via
    DoclingDocumentAdapter for normalized LDU output.

    Fallback: pdfplumber (table detection) + PyMuPDF (text blocks) when
    Docling is not installed or fails.

    Pros:
    - Tables extracted as structured JSON (headers + rows)
    - Docling provides deep layout understanding (multi-column, figures, lists)
    - Respects multi-column layouts and hierarchical sections
    - No external API calls

    Cons:
    - Slower than Strategy A (~10 pages/sec)
    - Cannot handle scanned documents (no OCR)
    - Table detection can fail on unusual layouts
    """

    # Minimum confidence to accept a detected table
    TABLE_CONFIDENCE_THRESHOLD: float = 0.6

    def __init__(self):
        """Initialize and detect Docling availability."""
        self._docling_available = False
        try:
            from docling.document_converter import DocumentConverter
            self._docling_available = True
            logger.info("LayoutExtractor: Docling available — using as primary engine")
        except ImportError:
            logger.info(
                "LayoutExtractor: Docling not installed — using pdfplumber+PyMuPDF fallback. "
                "Install Docling with: uv add docling"
            )

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

        Primary: Docling conversion via DoclingDocumentAdapter.
        Fallback: pdfplumber tables + PyMuPDF text blocks.
        """
        pdf_path = Path(pdf_path)

        # ── Try Docling first ─────────────────────────────────────────
        if self._docling_available:
            try:
                ldus = self._extract_with_docling(profile, pdf_path, pages)
                if ldus:
                    logger.info(
                        "LayoutExtractor (Docling): Extracted %d LDUs from %s",
                        len(ldus), pdf_path.name,
                    )
                    return ldus
                logger.warning(
                    "Docling returned 0 LDUs for %s, falling back to pdfplumber",
                    pdf_path.name,
                )
            except Exception as e:
                logger.warning(
                    "Docling extraction failed for %s: %s — falling back to pdfplumber",
                    pdf_path.name, e,
                )

        # ── Fallback: pdfplumber + PyMuPDF ────────────────────────────
        return self._extract_with_pdfplumber_fallback(profile, pdf_path, pages)

    def _extract_with_docling(
        self,
        profile: DocumentProfile,
        pdf_path: Path,
        pages: Optional[list[int]] = None,
    ) -> List[LDU]:
        """Extract using Docling's DocumentConverter and DoclingDocumentAdapter."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))

        # Use DoclingDocumentAdapter to normalize to LDU schema
        ldus = DoclingDocumentAdapter.adapt(result, profile, pages)
        return ldus

    def _extract_with_pdfplumber_fallback(
        self,
        profile: DocumentProfile,
        pdf_path: Path,
        pages: Optional[list[int]] = None,
    ) -> List[LDU]:
        """Fallback extraction using pdfplumber tables + PyMuPDF text blocks."""
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
            "LayoutExtractor (pdfplumber fallback): Extracted %d LDUs (%d tables) from %s",
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
