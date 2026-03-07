"""PDF analysis utilities for the Document Intelligence Refinery.

Provides lightweight analysis functions used by the Triage Agent to
classify documents without full extraction. All functions operate on
the PDF metadata and sampled pages only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def analyze_pdf_with_pdfplumber(pdf_path: str | Path) -> Dict[str, Any]:
    """Analyze a PDF using pdfplumber and return structural statistics.

    Returns:
        Dict with: page_count, total_chars, avg_char_density,
        image_count, image_to_page_ratio, has_text, fonts_found,
        pages_with_text, pages_with_images.
    """
    import pdfplumber

    stats: Dict[str, Any] = {
        "page_count": 0,
        "total_chars": 0,
        "avg_char_density": 0.0,
        "image_count": 0,
        "image_to_page_ratio": 0.0,
        "has_text": False,
        "fonts_found": [],
        "pages_with_text": 0,
        "pages_with_images": 0,
    }

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            stats["page_count"] = len(pdf.pages)
            total_chars = 0
            fonts = set()

            for page in pdf.pages:
                text = page.extract_text() or ""
                char_count = len(text.strip())
                total_chars += char_count

                if char_count > 10:
                    stats["pages_with_text"] += 1

                images = page.images or []
                if images:
                    stats["pages_with_images"] += 1
                    stats["image_count"] += len(images)

                # Collect font names
                for char_obj in (page.chars or [])[:50]:
                    font = char_obj.get("fontname", "")
                    if font:
                        fonts.add(font)

            stats["total_chars"] = total_chars
            stats["has_text"] = total_chars > 0
            stats["fonts_found"] = list(fonts)[:20]

            if stats["page_count"] > 0:
                stats["avg_char_density"] = total_chars / stats["page_count"]
                stats["image_to_page_ratio"] = (
                    stats["image_count"] / stats["page_count"]
                )

    except Exception as e:
        logger.error("PDF analysis failed for %s: %s", pdf_path, e)

    return stats


def detect_origin_type(stats: Dict[str, Any]) -> str:
    """Determine origin type from PDF statistics.

    Returns:
        One of: 'native_digital', 'scanned_image', 'mixed', 'form_fillable'.
    """
    page_count = stats.get("page_count", 0)
    if page_count == 0:
        return "scanned_image"

    text_coverage = stats.get("pages_with_text", 0) / max(page_count, 1)
    image_ratio = stats.get("image_to_page_ratio", 0)
    avg_chars = stats.get("avg_char_density", 0)

    # Scanned: very little text, many images
    if text_coverage < 0.2 and image_ratio > 0.5:
        return "scanned_image"

    # Native digital: good text coverage, few images
    if text_coverage > 0.7 and avg_chars > 100:
        return "native_digital"

    # Mixed: some text, some scanned
    if 0.2 <= text_coverage <= 0.7:
        return "mixed"

    return "native_digital"


def detect_layout_complexity(stats: Dict[str, Any]) -> str:
    """Determine layout complexity from PDF statistics.

    Returns:
        One of: 'simple', 'moderate', 'complex'.
    """
    image_ratio = stats.get("image_to_page_ratio", 0)
    fonts_count = len(stats.get("fonts_found", []))

    if image_ratio > 0.5 or fonts_count > 8:
        return "complex"
    elif image_ratio > 0.2 or fonts_count > 4:
        return "moderate"
    return "simple"


def detect_domain_hint(text: str) -> str:
    """Detect domain from sample text using keyword matching.

    Returns:
        One of: 'financial_report', 'legal_audit', 'technical_assessment',
        'structured_data', 'unknown'.
    """
    text_lower = text.lower()

    domain_keywords = {
        "financial_report": [
            "annual report", "financial statement", "balance sheet",
            "income statement", "total assets", "revenue", "dividend",
            "shareholders", "profit and loss", "fiscal year",
        ],
        "legal_audit": [
            "auditor", "audit report", "compliance", "regulation",
            "proclamation", "independent auditor", "legal opinion",
        ],
        "technical_assessment": [
            "assessment", "survey", "methodology", "evaluation",
            "recommendation", "performance indicator", "framework",
        ],
        "structured_data": [
            "tax expenditure", "customs", "tariff", "consumer price",
            "statistical", "fiscal data", "import tax",
        ],
    }

    scores = {}
    for domain, keywords in domain_keywords.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "unknown"
    return best


def estimate_extraction_cost(stats: Dict[str, Any], strategy: str) -> float:
    """Estimate extraction cost in USD.

    Args:
        stats: PDF analysis statistics.
        strategy: Strategy name ('fast_text', 'layout_aware', 'vision_model').

    Returns:
        Estimated cost in USD.
    """
    page_count = stats.get("page_count", 1)
    cost_per_page = {
        "fast_text": 0.0001,
        "layout_aware": 0.001,
        "vision_model": 0.01,
    }
    return page_count * cost_per_page.get(strategy, 0.001)


def extract_first_page_text(pdf_path: str | Path, max_pages: int = 3) -> str:
    """Extract text from the first N pages for classification.

    Args:
        pdf_path: Path to PDF file.
        max_pages: Number of pages to sample.

    Returns:
        Combined text from the first N pages.
    """
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        texts = []
        for i in range(min(max_pages, len(doc))):
            texts.append(doc[i].get_text("text"))
        doc.close()
        return "\n".join(texts)
    except Exception as e:
        logger.warning("Failed to extract first page text: %s", e)
        return ""
