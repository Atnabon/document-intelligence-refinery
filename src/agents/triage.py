"""Triage Agent — Phase 1 of the Document Intelligence Refinery.

Responsibilities:
1. Detect origin_type (native digital vs. scanned vs. mixed)
2. Assess layout_complexity (simple / moderate / complex)
3. Classify domain_hint (financial_report / legal_audit / technical_assessment / structured_data)
4. Recommend extraction strategy (Strategy A / B / C)

The Triage Agent operates WITHOUT extracting content — it analyses PDF metadata,
samples pages, and uses heuristics + optional LLM classification to produce a
DocumentProfile quickly and cheaply.

Domain classification is implemented as a swappable strategy: swap in a
VLMDomainClassifier (or any DomainClassifier subclass) without touching this
agent's core logic.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from src.models.document_profile import (
    DocumentProfile,
    DomainHint,
    ExtractionStrategy,
    LayoutComplexity,
    OriginType,
)

logger = logging.getLogger(__name__)


# ── Domain Classifier Strategy Interface ─────────────────────────────────────

class DomainClassifier(ABC):
    """Abstract base for domain classification strategies.

    Implement this interface to swap in a VLM-based classifier, an embedding
    classifier, or any other approach — without modifying TriageAgent.
    """

    @abstractmethod
    def classify(self, sample_text: str, filename: str) -> DomainHint:
        """Classify the document domain from sample text and filename.

        Args:
            sample_text: Text sampled from the first several pages.
            filename: Original filename (strong signal for some domains).

        Returns:
            DomainHint enum value.
        """
        ...


class KeywordDomainClassifier(DomainClassifier):
    """Keyword-scoring domain classifier (default, zero-cost).

    Scores each domain by counting keyword hits in the sample text and
    applying filename bonuses. Domain keyword lists are loaded from the
    externalized configuration at construction time so new domains can be
    onboarded purely by editing extraction_rules.yaml.
    """

    # Default keyword lists (used when config is not provided)
    _DEFAULT_KEYWORDS: Dict[DomainHint, List[str]] = {
        DomainHint.FINANCIAL_REPORT: [
            "annual report", "financial statement", "balance sheet",
            "income statement", "profit and loss", "total assets",
            "shareholders", "dividend", "fiscal year", "revenue",
            "net income", "bank", "capital adequacy",
        ],
        DomainHint.LEGAL_AUDIT: [
            "auditor", "audit report", "independent auditor",
            "legal opinion", "compliance", "regulation",
            "proclamation", "court", "judgment",
        ],
        DomainHint.TECHNICAL_ASSESSMENT: [
            "assessment", "survey", "methodology", "findings",
            "recommendation", "evaluation", "implementation",
            "performance", "indicator", "framework",
        ],
        DomainHint.STRUCTURED_DATA: [
            "tax expenditure", "import tax", "customs",
            "tariff", "consumer price index", "CPI",
            "statistical", "fiscal data", "expenditure",
        ],
    }

    _DEFAULT_FILENAME_PATTERNS: Dict[DomainHint, List[str]] = {
        DomainHint.FINANCIAL_REPORT: ["annual", "report"],
        DomainHint.LEGAL_AUDIT: ["audit"],
        DomainHint.TECHNICAL_ASSESSMENT: ["assessment", "survey"],
        DomainHint.STRUCTURED_DATA: ["tax", "expenditure", "cpi"],
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        """Load keyword lists from config or fall back to defaults."""
        domain_cfg = (config or {}).get("domain_classification", {})

        self._keywords: Dict[DomainHint, List[str]] = {}
        self._filename_patterns: Dict[DomainHint, List[str]] = {}

        domain_map = {
            "financial_report": DomainHint.FINANCIAL_REPORT,
            "legal_audit": DomainHint.LEGAL_AUDIT,
            "technical_assessment": DomainHint.TECHNICAL_ASSESSMENT,
            "structured_data": DomainHint.STRUCTURED_DATA,
        }

        for key, hint in domain_map.items():
            if key in domain_cfg:
                self._keywords[hint] = domain_cfg[key].get(
                    "keywords", self._DEFAULT_KEYWORDS[hint]
                )
                self._filename_patterns[hint] = domain_cfg[key].get(
                    "filename_patterns", self._DEFAULT_FILENAME_PATTERNS[hint]
                )
            else:
                self._keywords[hint] = self._DEFAULT_KEYWORDS[hint]
                self._filename_patterns[hint] = self._DEFAULT_FILENAME_PATTERNS[hint]

    def classify(self, sample_text: str, filename: str) -> DomainHint:
        text_lower = sample_text.lower()
        filename_lower = filename.lower()

        scores: Dict[DomainHint, int] = {hint: 0 for hint in self._keywords}

        # Keyword scoring from content
        for hint, keywords in self._keywords.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    scores[hint] += 1

        # Filename bonus (strong signal)
        for hint, patterns in self._filename_patterns.items():
            for pattern in patterns:
                if pattern.lower() in filename_lower:
                    scores[hint] += 3

        max_score = max(scores.values())
        if max_score == 0:
            return DomainHint.UNKNOWN

        return max(scores, key=scores.get)


class VLMDomainClassifier(DomainClassifier):
    """VLM-backed domain classifier — drop-in replacement for KeywordDomainClassifier.

    Pass an instance of this class to TriageAgent to upgrade domain classification
    to GPT-4o / Gemini without any changes to TriageAgent code.

    Usage:
        triage = TriageAgent(domain_classifier=VLMDomainClassifier())
    """

    def classify(self, sample_text: str, filename: str) -> DomainHint:
        """Classify using a VLM prompt (requires OPENAI_API_KEY)."""
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "Classify the document domain into exactly one of: "
                "financial_report, legal_audit, technical_assessment, structured_data, unknown.\n\n"
                f"Filename: {filename}\n\nSample text:\n{sample_text[:2000]}\n\n"
                "Respond with only the domain label, nothing else."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            label = response.choices[0].message.content.strip().lower()
            domain_map = {
                "financial_report": DomainHint.FINANCIAL_REPORT,
                "legal_audit": DomainHint.LEGAL_AUDIT,
                "technical_assessment": DomainHint.TECHNICAL_ASSESSMENT,
                "structured_data": DomainHint.STRUCTURED_DATA,
            }
            return domain_map.get(label, DomainHint.UNKNOWN)
        except Exception as e:
            logger.warning("VLMDomainClassifier failed, returning UNKNOWN: %s", e)
            return DomainHint.UNKNOWN


# ── Triage Agent ──────────────────────────────────────────────────────────────


class TriageAgent:
    """Classifies incoming PDFs and produces a DocumentProfile.

    Design principles:
    - Fast: should complete triage in < 2 seconds for a 200-page PDF.
    - Cheap: no LLM calls in the default path; LLM used only for ambiguous cases.
    - Deterministic: same document always produces the same profile.

    The domain classifier is injected as a strategy: pass a VLMDomainClassifier
    instance to upgrade classification without modifying this class.
    """

    # ── Defaults (overridden by extraction_rules.yaml via config) ──────
    SCANNED_TEXT_THRESHOLD: int = 50
    SCANNED_RATIO_THRESHOLD: float = 0.5
    MIXED_RATIO_LOWER: float = 0.1
    TABLE_KEYWORD_THRESHOLD: int = 3
    COMPLEX_TABLE_THRESHOLD: int = 5

    TABLE_KEYWORDS = [
        "table", "total", "amount", "percentage", "%",
        "sum", "average", "row", "column",
    ]

    def __init__(
        self,
        config: Optional[dict] = None,
        domain_classifier: Optional[DomainClassifier] = None,
    ) -> None:
        """Initialize with optional configuration overrides and domain classifier.

        Args:
            config: Dictionary from extraction_rules.yaml (triage section).
            domain_classifier: A DomainClassifier implementation. Defaults to
                               KeywordDomainClassifier loaded from config.
        """
        triage_cfg = (config or {}).get("triage", config or {})

        self.SCANNED_TEXT_THRESHOLD = triage_cfg.get(
            "scanned_text_threshold", self.SCANNED_TEXT_THRESHOLD
        )
        self.SCANNED_RATIO_THRESHOLD = triage_cfg.get(
            "scanned_ratio_threshold", self.SCANNED_RATIO_THRESHOLD
        )
        self.MIXED_RATIO_LOWER = triage_cfg.get(
            "mixed_ratio_lower", self.MIXED_RATIO_LOWER
        )
        self.TABLE_KEYWORD_THRESHOLD = triage_cfg.get(
            "table_keyword_threshold", self.TABLE_KEYWORD_THRESHOLD
        )
        self.COMPLEX_TABLE_THRESHOLD = triage_cfg.get(
            "complex_table_threshold", self.COMPLEX_TABLE_THRESHOLD
        )

        # Domain classifier strategy — inject VLMDomainClassifier to upgrade
        if domain_classifier is not None:
            self._domain_classifier = domain_classifier
        else:
            self._domain_classifier = KeywordDomainClassifier(
                config=triage_cfg.get("domain_classification")
                and {"domain_classification": triage_cfg["domain_classification"]}
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
        domain_hint = self._domain_classifier.classify(sample_text, pdf_path.name)

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
