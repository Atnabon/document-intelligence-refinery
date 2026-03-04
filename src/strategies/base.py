"""Base extraction strategy interface.

All extraction strategies implement this abstract base class to ensure a
consistent interface for the ExtractionRouter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.models.document_profile import DocumentProfile
from src.models.ldu import LDU


class BaseExtractor(ABC):
    """Abstract base class for all extraction strategies.

    Subclasses must implement `extract()` which takes a DocumentProfile and
    PDF path and returns a list of LDUs.
    """

    @abstractmethod
    def extract(
        self,
        profile: DocumentProfile,
        pdf_path: str | Path,
        pages: list[int] | None = None,
    ) -> List[LDU]:
        """Extract LDUs from a PDF.

        Args:
            profile: The DocumentProfile from triage.
            pdf_path: Path to the source PDF.
            pages: Optional list of specific page numbers (1-based) to extract.
                   If None, extract all pages.

        Returns:
            List of extracted LDUs.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the strategy name identifier."""
        ...

    @abstractmethod
    def cost_per_page(self) -> float:
        """Return the estimated cost per page in USD."""
        ...
