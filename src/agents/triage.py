"""Triage Agent — Phase 1 of the Document Intelligence Refinery.

Responsibilities:
1. Detect origin_type (native digital vs. scanned vs. mixed)
2. Assess layout_complexity (simple / moderate / complex)
3. Classify domain_hint (financial_report / legal_audit / technical_assessment / structured_data)
4. Recommend extraction strategy (Strategy A / B / C)

The Triage Agent operates WITHOUT extracting content — it analyses PDF metadata,
samples pages, and uses heuristics + optional LLM classification to produce a
DocumentProfile quickly and cheaply.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.models.document_profile import (
    DocumentProfile,
    DomainHint,
    ExtractionStrategy,
    LayoutComplexity,
    OriginType,
)

logger = logging.getLogger(__name__)


class TriageAgent:
    """Classifies incoming PDFs and produces a DocumentProfile.

    Design principles:
    - Fast: should complete triage in < 2 seconds for a 200-page PDF.
    - Cheap: no LLM calls in the default path; LLM used only for ambiguous cases.
    - Deterministic: same document always produces the same profile.
    """

    # ── Thresholds (configurable via extraction_rules.yaml) ────────────
    SCANNED_TEXT_THRESHOLD: int = 50  # chars per page below which → scanned
    SCANNED_RATIO_THRESHOLD: float = 0.5  # above this → SCANNED_IMAGE
    MIXED_RATIO_LOWER: float = 0.1  # below this → NATIVE_DIGITAL
    TABLE_KEYWORD_THRESHOLD: int = 3  # keyword hits to flag has_tables
    COMPLEX_TABLE_THRESHOLD: int = 5  # tables count for COMPLEX layout

    # Keywords that hint at domain classification
    FINANCIAL_KEYWORDS = [
        "annual report", "financial statement", "balance sheet",
        "income statement", "profit and loss", "total assets",
        "shareholders", "dividend", "fiscal year", "revenue",
        "net income", "bank", "capital adequacy",
    ]
    LEGAL_KEYWORDS = [
        "auditor", "audit report", "independent auditor",
        "legal opinion", "compliance", "regulation",
        "proclamation", "court", "judgment",
    ]
    TECHNICAL_KEYWORDS = [
        "assessment", "survey", "methodology", "findings",
        "recommendation", "evaluation", "implementation",
        "performance", "indicator", "framework",
    ]
    STRUCTURED_DATA_KEYWORDS = [
        "tax expenditure", "import tax", "customs",
        "tariff", "consumer price index", "CPI",
        "statistical", "fiscal data", "expenditure",
    ]
    TABLE_KEYWORDS = [
        "table", "total", "amount", "percentage", "%",
        "sum", "average", "row", "column",
    ]

    def __init__(self, config: Optional[dict] = None):
        """Initialize with optional configuration overrides."""
        if config:
            self.SCANNED_TEXT_THRESHOLD = config.get(
                "scanned_text_threshold", self.SCANNED_TEXT_THRESHOLD
            )
            self.SCANNED_RATIO_THRESHOLD = config.get(
                "scanned_ratio_threshold", self.SCANNED_RATIO_THRESHOLD
            )

    def profile_document(self, pdf_path: str | Path) -> DocumentProfile:
        """Analyse a PDF and produce a DocumentProfile.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Fully populated DocumentProfile.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If the file is not a valid PDF.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Triaging document: %s", pdf_path.name)

        # Compute file hash
        file_hash = self._compute_file_hash(pdf_path)

        # Open PDF
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)

        # ── Phase 1: Origin Type Detection ────────────────────────────
        origin_type, scanned_ratio = self._detect_origin_type(doc)

        # ── Phase 2: Layout Complexity Detection ──────────────────────
        layout_complexity, has_tables, has_images, has_footnotes = (
            self._assess_layout(doc, origin_type)
        )

        # ── Phase 3: Domain Classification ────────────────────────────
        sample_text = self._extract_sample_text(doc)
        domain_hint = self._classify_domain(sample_text, pdf_path.name)

        # ── Phase 4: Strategy Selection ───────────────────────────────
        strategy, rationale, cost = self._select_strategy(
            origin_type, layout_complexity, domain_hint, page_count, has_tables
        )

        # ── Phase 5: Detect language ──────────────────────────────────
        language = self._detect_language(sample_text)

        doc.close()

        document_id = self._make_document_id(pdf_path.stem)

        return DocumentProfile(
            document_id=document_id,
            filename=pdf_path.name,
            file_hash=file_hash,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            domain_hint=domain_hint,
            page_count=page_count,
            language=language,
            has_tables=has_tables,
            has_images=has_images,
            has_footnotes=has_footnotes,
            scanned_page_ratio=scanned_ratio,
            recommended_strategy=strategy,
            strategy_rationale=rationale,
            estimated_cost_usd=cost,
            profiled_at=datetime.utcnow(),
        )

    # ── Internal Methods ─────────────────────────────────────────────────

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """SHA-256 hash of the file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _detect_origin_type(
        self, doc: fitz.Document
    ) -> tuple[OriginType, float]:
        """Determine whether pages are native digital or scanned.

        Heuristic: extract text from each page. If char count < threshold,
        the page is likely a scanned image.
        """
        scanned_pages = 0
        total = len(doc)

        # Sample up to 20 pages evenly across the document
        sample_indices = self._sample_page_indices(total, max_samples=20)

        for idx in sample_indices:
            page = doc[idx]
            text = page.get_text("text")
            if len(text.strip()) < self.SCANNED_TEXT_THRESHOLD:
                scanned_pages += 1

        scanned_ratio = scanned_pages / len(sample_indices) if sample_indices else 0.0

        if scanned_ratio >= self.SCANNED_RATIO_THRESHOLD:
            origin = OriginType.SCANNED_IMAGE
        elif scanned_ratio >= self.MIXED_RATIO_LOWER:
            origin = OriginType.MIXED
        else:
            origin = OriginType.NATIVE_DIGITAL

        return origin, round(scanned_ratio, 3)

    def _assess_layout(
        self, doc: fitz.Document, origin_type: OriginType
    ) -> tuple[LayoutComplexity, bool, bool, bool]:
        """Assess layout complexity and detect structural elements."""
        has_tables = False
        has_images = False
        has_footnotes = False
        table_count = 0
        multi_column_count = 0

        sample_indices = self._sample_page_indices(len(doc), max_samples=15)

        for idx in sample_indices:
            page = doc[idx]

            # Check for images
            image_list = page.get_images(full=True)
            if image_list:
                has_images = True

            # Check for tables via text heuristics (looking for grid patterns)
            text = page.get_text("text")
            table_kw_count = sum(
                1 for kw in self.TABLE_KEYWORDS if kw.lower() in text.lower()
            )
            if table_kw_count >= self.TABLE_KEYWORD_THRESHOLD:
                has_tables = True
                table_count += 1

            # Check for footnotes
            if re.search(r"\b\d+\s*\.\s+[A-Z]", text) or "footnote" in text.lower():
                has_footnotes = True

            # Rough multi-column detection via text block analysis
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            if blocks and "blocks" in blocks:
                x_positions = set()
                for block in blocks["blocks"]:
                    if block.get("type") == 0:  # text block
                        x_positions.add(round(block["bbox"][0] / 50) * 50)
                if len(x_positions) > 2:
                    multi_column_count += 1

        # Determine complexity
        multi_column_ratio = (
            multi_column_count / len(sample_indices) if sample_indices else 0
        )

        if table_count >= self.COMPLEX_TABLE_THRESHOLD or (
            multi_column_ratio > 0.3 and has_tables
        ):
            complexity = LayoutComplexity.COMPLEX
        elif has_tables or multi_column_ratio > 0.2 or has_images:
            complexity = LayoutComplexity.MODERATE
        else:
            complexity = LayoutComplexity.SIMPLE

        return complexity, has_tables, has_images, has_footnotes

    def _extract_sample_text(self, doc: fitz.Document, max_chars: int = 5000) -> str:
        """Extract sample text from the first few pages for classification."""
        texts = []
        char_count = 0
        for i in range(min(10, len(doc))):
            text = doc[i].get_text("text")
            texts.append(text)
            char_count += len(text)
            if char_count >= max_chars:
                break
        return "\n".join(texts)[:max_chars]

    def _classify_domain(self, sample_text: str, filename: str) -> DomainHint:
        """Classify document domain using keyword matching.

        Uses a weighted scoring system across domain keyword sets.
        Filename is also considered as a strong signal.
        """
        text_lower = sample_text.lower()
        filename_lower = filename.lower()

        scores = {
            DomainHint.FINANCIAL_REPORT: 0,
            DomainHint.LEGAL_AUDIT: 0,
            DomainHint.TECHNICAL_ASSESSMENT: 0,
            DomainHint.STRUCTURED_DATA: 0,
        }

        # Score from text content
        for kw in self.FINANCIAL_KEYWORDS:
            if kw in text_lower:
                scores[DomainHint.FINANCIAL_REPORT] += 1
        for kw in self.LEGAL_KEYWORDS:
            if kw in text_lower:
                scores[DomainHint.LEGAL_AUDIT] += 1
        for kw in self.TECHNICAL_KEYWORDS:
            if kw in text_lower:
                scores[DomainHint.TECHNICAL_ASSESSMENT] += 1
        for kw in self.STRUCTURED_DATA_KEYWORDS:
            if kw in text_lower:
                scores[DomainHint.STRUCTURED_DATA] += 1

        # Filename bonus (strong signal)
        if "annual" in filename_lower or "report" in filename_lower:
            scores[DomainHint.FINANCIAL_REPORT] += 3
        if "audit" in filename_lower:
            scores[DomainHint.LEGAL_AUDIT] += 3
        if "assessment" in filename_lower or "survey" in filename_lower:
            scores[DomainHint.TECHNICAL_ASSESSMENT] += 3
        if "tax" in filename_lower or "expenditure" in filename_lower or "cpi" in filename_lower:
            scores[DomainHint.STRUCTURED_DATA] += 3

        max_score = max(scores.values())
        if max_score == 0:
            return DomainHint.UNKNOWN

        # Return highest-scoring domain
        return max(scores, key=scores.get)

    def _select_strategy(
        self,
        origin: OriginType,
        complexity: LayoutComplexity,
        domain: DomainHint,
        page_count: int,
        has_tables: bool,
    ) -> tuple[ExtractionStrategy, str, float]:
        """Select extraction strategy based on document characteristics.

        Decision Tree:
        1. Scanned documents → Strategy C (VLM) — no text stream available
        2. Complex layout with tables → Strategy B (Layout-aware)
        3. Simple native digital → Strategy A (Fast text)
        4. Mixed documents → Strategy B (with possible escalation to C)

        Cost model (per page):
        - Strategy A: ~$0.0001 (local processing only)
        - Strategy B: ~$0.001  (layout analysis + table extraction)
        - Strategy C: ~$0.01   (VLM API call per page)
        """
        per_page_cost = {
            ExtractionStrategy.STRATEGY_A: 0.0001,
            ExtractionStrategy.STRATEGY_B: 0.001,
            ExtractionStrategy.STRATEGY_C: 0.01,
        }

        # Decision tree
        if origin == OriginType.SCANNED_IMAGE:
            strategy = ExtractionStrategy.STRATEGY_C
            rationale = (
                "Document is scanned (no character stream). "
                "VLM-based extraction required for OCR and layout understanding."
            )
        elif origin == OriginType.MIXED:
            strategy = ExtractionStrategy.STRATEGY_B
            rationale = (
                "Mixed document (some pages scanned). "
                "Layout-aware extraction with per-page escalation to VLM for scanned pages."
            )
        elif complexity == LayoutComplexity.COMPLEX and has_tables:
            strategy = ExtractionStrategy.STRATEGY_B
            rationale = (
                "Native digital but complex layout with heavy tables. "
                "Layout-aware extraction needed for table fidelity."
            )
        elif complexity == LayoutComplexity.MODERATE:
            if has_tables and domain in (
                DomainHint.FINANCIAL_REPORT,
                DomainHint.STRUCTURED_DATA,
            ):
                strategy = ExtractionStrategy.STRATEGY_B
                rationale = (
                    f"Moderate complexity with tables in {domain} domain. "
                    "Layout-aware extraction preferred for numerical accuracy."
                )
            else:
                strategy = ExtractionStrategy.STRATEGY_A
                rationale = (
                    "Moderate complexity but text-dominant. "
                    "Fast text extraction sufficient."
                )
        else:
            strategy = ExtractionStrategy.STRATEGY_A
            rationale = (
                "Simple native digital layout. "
                "Fast text extraction is optimal for speed and cost."
            )

        cost = per_page_cost[strategy] * page_count
        return strategy, rationale, round(cost, 4)

    @staticmethod
    def _detect_language(text: str) -> str:
        """Simple language detection heuristic."""
        # Basic: check for Amharic Unicode range
        amharic_chars = sum(1 for c in text if "\u1200" <= c <= "\u137F")
        if amharic_chars > len(text) * 0.1:
            return "am"
        return "en"

    @staticmethod
    def _make_document_id(stem: str) -> str:
        """Create a clean document ID from filename stem."""
        # Normalize: lowercase, replace spaces/special chars with underscores
        doc_id = re.sub(r"[^a-zA-Z0-9]+", "_", stem.lower()).strip("_")
        return doc_id

    @staticmethod
    def _sample_page_indices(total: int, max_samples: int = 20) -> list[int]:
        """Get evenly-spaced sample page indices."""
        if total <= max_samples:
            return list(range(total))
        step = total / max_samples
        return [int(i * step) for i in range(max_samples)]
