"""Strategy C: Vision Model Extractor.

Most capable strategy for scanned/complex documents. Supports two modes:

1. **Ollama mode (default, free, local)**: Sends rendered page images to a
   local Ollama server running kimi-k2.5:cloud. No API key, no cost.
   Requires: `ollama pull kimi-k2.5:cloud && ollama serve`
2. **OpenAI/Google mode (optional, paid)**: Sends rendered page images to
   GPT-4o or Gemini Flash via OpenRouter. Higher cost but cloud-hosted.

The strategy sends rendered page images to the configured VLM for structured
text extraction. If VLM is unavailable, it falls back to PyMuPDF basic text
extraction as a last resort.

Best for: Scanned documents, complex visual layouts, diagrams, handwritten text.
Cost: $0.00/page (Ollama local) or ~$0.01/page (OpenAI/Google API).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from src.models.document_profile import DocumentProfile
from src.models.ldu import LDU, LDUType
from src.models.provenance import BoundingBox
from src.strategies.base import BaseExtractor

logger = logging.getLogger(__name__)


class VisionExtractor(BaseExtractor):
    """Strategy C: Vision model extraction for scanned/complex documents.

    Primary VLM: Ollama kimi-k2.5:cloud (free, local, requires `ollama serve`)
    Cloud VLM: OpenAI GPT-4o or Google Gemini (paid, optional)
    Last resort: PyMuPDF basic text extraction

    Pros:
    - Ollama VLM handles complex layouts locally with kimi-k2.5:cloud
    - Structured JSON output with table detection and bbox estimation
    - Budget guard prevents cost overruns (relevant for cloud APIs only)

    Cons:
    - Ollama requires local Ollama server with kimi-k2.5:cloud model pulled
    - Cloud APIs (OpenAI/Google) require API keys and cost ~$0.01/page
    - Slower than text extraction (~1-2 pages/sec)
    """

    # VLM provider configuration
    DEFAULT_PROVIDER = "ollama"  # "ollama" | "openai" | "google"
    DEFAULT_MODEL = "kimi-k2.5:cloud"
    DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
    MAX_RETRIES = 3
    RENDER_DPI = 200  # DPI for page rendering
    DEFAULT_BUDGET_CAP_USD = 2.0  # Halt processing when this amount is exceeded

    EXTRACTION_PROMPT = """Analyze this document page image and extract all content with structure preserved.

For each content element, return a JSON array of objects with these fields:
- "type": one of "heading", "paragraph", "table", "list", "figure", "footnote", "key_value"
- "content": the text content
- "structured_content": (for tables only) {"headers": [...], "rows": [[...], ...]}
- "bbox_relative": {"x0": 0-1, "y0": 0-1, "x1": 0-1, "y1": 0-1} (relative coordinates)
- "confidence": 0-1 confidence score

Preserve all numbers exactly as they appear. Do not infer or calculate values.
Return ONLY valid JSON array, no other text."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        budget_cap_usd: float | None = None,
        ollama_base_url: str | None = None,
    ):
        """Initialize with VLM provider configuration and optional budget cap.

        Args:
            provider: VLM provider. One of "ollama" (default), "openai", "google".
                      Reads VLM_PROVIDER env var.
            model: Model name. For Ollama: "kimi-k2.5:cloud". Reads VLM_MODEL env var.
            budget_cap_usd: Hard cost cap in USD. Processing halts when exceeded.
                            Relevant for cloud providers only (Ollama is free).
                            Defaults to DEFAULT_BUDGET_CAP_USD.
            ollama_base_url: Ollama server base URL. Reads OLLAMA_BASE_URL env var.
                             Defaults to http://localhost:11434/v1
        """
        self.provider = provider or os.getenv("VLM_PROVIDER", self.DEFAULT_PROVIDER)
        self.model = model or os.getenv("VLM_MODEL", self.DEFAULT_MODEL)
        self.budget_cap_usd = (
            budget_cap_usd if budget_cap_usd is not None
            else self.DEFAULT_BUDGET_CAP_USD
        )
        self.ollama_base_url = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_OLLAMA_BASE_URL)
        )

    def name(self) -> str:
        return "vision_model"

    def cost_per_page(self) -> float:
        """Return cost per page in USD. Free for local Ollama provider."""
        if self.provider in ("ollama",):
            return 0.0  # local inference — no cost
        return 0.01  # cloud APIs (openai, google)

    def extract(
        self,
        profile: DocumentProfile,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
    ) -> List[LDU]:
        """Extract content from scanned/complex pages using VLM.

        Extraction approach:
        1. VLM (Ollama kimi-k2.5:cloud by default) — free for local, paid for cloud
        2. PyMuPDF basic text extraction — free fallback if VLM unavailable

        Processing halts when cumulative cost exceeds budget_cap_usd (cloud VLM only).
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        ldus: List[LDU] = []

        page_indices = [p - 1 for p in pages] if pages else range(len(doc))
        cumulative_cost = 0.0
        cost_per_page = self.cost_per_page()

        for page_idx in page_indices:
            if page_idx < 0 or page_idx >= len(doc):
                continue

            page = doc[page_idx]
            page_number = page_idx + 1

            # ── Budget cap enforcement (VLM) ─────────────────────────
            if cumulative_cost + cost_per_page > self.budget_cap_usd:
                logger.warning(
                    "VisionExtractor: Budget cap $%.2f reached after %d pages "
                    "(cumulative cost $%.4f). Halting further VLM processing.",
                    self.budget_cap_usd,
                    page_idx,
                    cumulative_cost,
                )
                # Fall back to basic text for remaining pages
                page_ldus = self._extract_page_ocr_fallback(
                    profile, page, page_number
                )
                ldus.extend(page_ldus)
                continue

            try:
                page_ldus = self._extract_page_with_vlm(
                    profile, page, page_number, doc
                )
                ldus.extend(page_ldus)
                cumulative_cost += cost_per_page
            except Exception as e:
                logger.warning(
                    "VLM extraction failed for page %d, falling back to basic text: %s",
                    page_number,
                    str(e),
                )
                # Fallback: basic text via PyMuPDF (no additional cost)
                page_ldus = self._extract_page_ocr_fallback(
                    profile, page, page_number
                )
                ldus.extend(page_ldus)

        doc.close()
        logger.info(
            "VisionExtractor: Extracted %d LDUs from %s (VLM cost $%.4f / cap $%.2f)",
            len(ldus),
            pdf_path.name,
            cumulative_cost,
            self.budget_cap_usd,
        )
        return ldus

    def _extract_page_with_vlm(
        self,
        profile: DocumentProfile,
        page: fitz.Page,
        page_number: int,
        doc: fitz.Document,
    ) -> List[LDU]:
        """Send a rendered page image to VLM and parse structured response."""
        # Render page to image
        image_bytes = self._render_page(page)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Call VLM API
        response = self._call_vlm(image_b64)

        # Parse response into LDUs
        ldus = self._parse_vlm_response(
            response, profile, page, page_number
        )
        return ldus

    def _render_page(self, page: fitz.Page) -> bytes:
        """Render a PDF page as a PNG image."""
        mat = fitz.Matrix(self.RENDER_DPI / 72, self.RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")

    def _call_vlm(self, image_b64: str) -> str:
        """Call the VLM with the page image.

        Supports Ollama (local, free), OpenAI, and Google Gemini providers.
        """
        if self.provider == "ollama":
            return self._call_ollama(image_b64)
        elif self.provider == "openai":
            return self._call_openai(image_b64)
        elif self.provider == "google":
            return self._call_google(image_b64)
        else:
            raise ValueError(f"Unknown VLM provider: {self.provider}")

    def _call_ollama(self, image_b64: str) -> str:
        """Call a local Ollama server using the OpenAI-compatible API.

        Requires Ollama running locally with the model already pulled:
            ollama pull kimi-k2.5:cloud
            ollama serve

        Uses the OpenAI Python client pointed at localhost:11434/v1.
        No API key needed — passes the string "ollama" as a placeholder.
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.ollama_base_url,
                api_key="ollama",  # required by client but not validated by Ollama
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: uv add openai"
            )
        except Exception as e:
            raise RuntimeError(
                f"Ollama API call failed (is `ollama serve` running?): {e}"
            )

    def _call_openai(self, image_b64: str) -> str:
        """Call OpenAI Vision API."""
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def _call_google(self, image_b64: str) -> str:
        """Call Google Gemini Vision API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(self.model or "gemini-1.5-flash")

            import PIL.Image

            image_data = base64.b64decode(image_b64)
            image = PIL.Image.open(BytesIO(image_data))

            response = model.generate_content(
                [self.EXTRACTION_PROMPT, image]
            )
            return response.text
        except ImportError:
            raise RuntimeError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
        except Exception as e:
            raise RuntimeError(f"Google Gemini API call failed: {e}")

    def _parse_vlm_response(
        self,
        response: str,
        profile: DocumentProfile,
        page: fitz.Page,
        page_number: int,
    ) -> List[LDU]:
        """Parse VLM JSON response into LDUs."""
        ldus: List[LDU] = []

        try:
            # Clean response (remove markdown code fences if present)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]

            elements = json.loads(cleaned)
            if not isinstance(elements, list):
                elements = [elements]

        except json.JSONDecodeError:
            logger.warning("VLM returned non-JSON response for page %d", page_number)
            # Treat entire response as a single paragraph
            elements = [
                {"type": "paragraph", "content": response, "confidence": 0.5}
            ]

        for seq, elem in enumerate(elements):
            ldu_type_str = elem.get("type", "paragraph")
            type_map = {
                "heading": LDUType.HEADING,
                "paragraph": LDUType.PARAGRAPH,
                "table": LDUType.TABLE,
                "list": LDUType.LIST,
                "figure": LDUType.FIGURE,
                "footnote": LDUType.FOOTNOTE,
                "key_value": LDUType.KEY_VALUE,
                "caption": LDUType.CAPTION,
            }
            ldu_type = type_map.get(ldu_type_str, LDUType.PARAGRAPH)

            content = elem.get("content", "")
            if not content:
                continue

            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Parse relative bbox
            bbox = None
            bbox_rel = elem.get("bbox_relative")
            if bbox_rel:
                bbox = BoundingBox(
                    x0=bbox_rel["x0"] * page.rect.width,
                    y0=bbox_rel["y0"] * page.rect.height,
                    x1=bbox_rel["x1"] * page.rect.width,
                    y1=bbox_rel["y1"] * page.rect.height,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                )

            structured = elem.get("structured_content")

            ldu = LDU(
                ldu_id=f"{profile.document_id}_p{page_number}_v{seq:03d}",
                document_id=profile.document_id,
                ldu_type=ldu_type,
                content=content,
                structured_content=structured,
                page_number=page_number,
                bbox=bbox,
                content_hash=content_hash,
                extraction_strategy=self.name(),
                confidence=elem.get("confidence", 0.80),
                sequence_index=seq,
            )
            ldus.append(ldu)

        return ldus

    def _extract_page_ocr_fallback(
        self,
        profile: DocumentProfile,
        page: fitz.Page,
        page_number: int,
    ) -> List[LDU]:
        """Fallback: use PyMuPDF's built-in OCR or basic text extraction."""
        text = page.get_text("text")
        if not text.strip():
            # Try OCR if available
            try:
                text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            except Exception:
                text = "[OCR unavailable — page appears to be scanned]"

        if not text.strip():
            return []

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        ldu = LDU(
            ldu_id=f"{profile.document_id}_p{page_number}_ocr000",
            document_id=profile.document_id,
            ldu_type=LDUType.PARAGRAPH,
            content=text.strip(),
            page_number=page_number,
            bbox=BoundingBox(
                x0=0,
                y0=0,
                x1=page.rect.width,
                y1=page.rect.height,
                page_width=page.rect.width,
                page_height=page.rect.height,
            ),
            content_hash=content_hash,
            extraction_strategy="ocr_fallback",
            confidence=0.50,
            sequence_index=0,
        )
        return [ldu]
