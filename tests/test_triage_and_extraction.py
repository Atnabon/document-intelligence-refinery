"""Unit tests for the Triage Agent classification logic."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.triage import TriageAgent
from src.models.document_profile import (
    DocumentProfile,
    DomainHint,
    ExtractionStrategy,
    LayoutComplexity,
    OriginType,
)


class TestTriageAgentDomainClassification:
    """Test domain classification logic."""

    def setup_method(self):
        self.agent = TriageAgent()

    def test_classify_financial_report_by_content(self):
        """Financial keywords in text should yield FINANCIAL_REPORT."""
        text = (
            "Annual Report for Fiscal Year 2023-24. "
            "Total assets reached ETB 2.1 trillion. "
            "Net income increased by 15%. Balance sheet shows strong capital adequacy."
        )
        result = self.agent._classify_domain(text, "some_document.pdf")
        assert result == DomainHint.FINANCIAL_REPORT

    def test_classify_financial_report_by_filename(self):
        """Filename containing 'annual report' should boost financial classification."""
        text = "Some generic text about the organization."
        result = self.agent._classify_domain(text, "CBE Annual Report 2023-24.pdf")
        assert result == DomainHint.FINANCIAL_REPORT

    def test_classify_legal_audit_by_content(self):
        """Audit keywords should yield LEGAL_AUDIT."""
        text = (
            "Independent Auditor's Report. We have audited the accompanying "
            "financial statements. The audit report covers compliance with "
            "applicable regulations and proclamation requirements."
        )
        result = self.agent._classify_domain(text, "document.pdf")
        assert result == DomainHint.LEGAL_AUDIT

    def test_classify_legal_audit_by_filename(self):
        """Filename containing 'audit' should boost legal/audit classification."""
        text = "Independent auditor opinion on compliance and regulation."
        result = self.agent._classify_domain(text, "Audit Report - 2023.pdf")
        assert result == DomainHint.LEGAL_AUDIT

    def test_classify_technical_assessment(self):
        """Technical assessment keywords should yield TECHNICAL_ASSESSMENT."""
        text = (
            "Assessment of Financial Transparency and Accountability. "
            "This survey evaluates the implementation of performance indicators. "
            "The methodology uses a comprehensive evaluation framework."
        )
        result = self.agent._classify_domain(text, "fta_performance_survey.pdf")
        assert result == DomainHint.TECHNICAL_ASSESSMENT

    def test_classify_structured_data(self):
        """Structured data/tax keywords should yield STRUCTURED_DATA."""
        text = (
            "Tax Expenditure Report covering import tax data for Ethiopia. "
            "Consumer Price Index analysis. Customs tariff expenditure breakdown."
        )
        result = self.agent._classify_domain(text, "tax_expenditure_ethiopia.pdf")
        assert result == DomainHint.STRUCTURED_DATA

    def test_classify_unknown_domain(self):
        """Text with no domain keywords should yield UNKNOWN."""
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        result = self.agent._classify_domain(text, "random.pdf")
        assert result == DomainHint.UNKNOWN


class TestTriageAgentStrategySelection:
    """Test strategy selection decision tree."""

    def setup_method(self):
        self.agent = TriageAgent()

    def test_scanned_document_gets_vision_strategy(self):
        """Scanned documents should always get Strategy C (VLM)."""
        strategy, rationale, cost = self.agent._select_strategy(
            origin=OriginType.SCANNED_IMAGE,
            complexity=LayoutComplexity.SIMPLE,
            domain=DomainHint.LEGAL_AUDIT,
            page_count=50,
            has_tables=False,
        )
        assert strategy == ExtractionStrategy.STRATEGY_C
        assert "scanned" in rationale.lower() or "VLM" in rationale

    def test_mixed_document_gets_layout_strategy(self):
        """Mixed documents should get Strategy B (layout-aware)."""
        strategy, rationale, cost = self.agent._select_strategy(
            origin=OriginType.MIXED,
            complexity=LayoutComplexity.MODERATE,
            domain=DomainHint.FINANCIAL_REPORT,
            page_count=100,
            has_tables=True,
        )
        assert strategy == ExtractionStrategy.STRATEGY_B

    def test_complex_with_tables_gets_layout_strategy(self):
        """Complex native digital with tables → Strategy B."""
        strategy, rationale, cost = self.agent._select_strategy(
            origin=OriginType.NATIVE_DIGITAL,
            complexity=LayoutComplexity.COMPLEX,
            domain=DomainHint.FINANCIAL_REPORT,
            page_count=120,
            has_tables=True,
        )
        assert strategy == ExtractionStrategy.STRATEGY_B

    def test_simple_native_gets_fast_text(self):
        """Simple native digital → Strategy A (fast text)."""
        strategy, rationale, cost = self.agent._select_strategy(
            origin=OriginType.NATIVE_DIGITAL,
            complexity=LayoutComplexity.SIMPLE,
            domain=DomainHint.UNKNOWN,
            page_count=20,
            has_tables=False,
        )
        assert strategy == ExtractionStrategy.STRATEGY_A

    def test_cost_scales_with_page_count(self):
        """Cost should be proportional to page count."""
        _, _, cost_small = self.agent._select_strategy(
            origin=OriginType.NATIVE_DIGITAL,
            complexity=LayoutComplexity.SIMPLE,
            domain=DomainHint.UNKNOWN,
            page_count=10,
            has_tables=False,
        )
        _, _, cost_large = self.agent._select_strategy(
            origin=OriginType.NATIVE_DIGITAL,
            complexity=LayoutComplexity.SIMPLE,
            domain=DomainHint.UNKNOWN,
            page_count=100,
            has_tables=False,
        )
        assert cost_large > cost_small

    def test_vision_strategy_is_most_expensive(self):
        """VLM strategy should be the most expensive per page."""
        _, _, cost_a = self.agent._select_strategy(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.SIMPLE,
            DomainHint.UNKNOWN, 100, False,
        )
        _, _, cost_c = self.agent._select_strategy(
            OriginType.SCANNED_IMAGE, LayoutComplexity.SIMPLE,
            DomainHint.UNKNOWN, 100, False,
        )
        assert cost_c > cost_a


class TestTriageAgentOriginDetection:
    """Test origin type detection heuristics."""

    def setup_method(self):
        self.agent = TriageAgent()

    def test_language_detection_english(self):
        """English text should be detected as 'en'."""
        text = "This is a standard English language document."
        assert self.agent._detect_language(text) == "en"

    def test_language_detection_amharic(self):
        """Amharic text should be detected as 'am'."""
        # Simulate Amharic characters (Ethiopian script)
        text = "ኢትዮጵያ" * 20 + "abc"
        assert self.agent._detect_language(text) == "am"

    def test_make_document_id(self):
        """Document IDs should be clean, lowercase, underscore-separated."""
        assert self.agent._make_document_id("CBE Annual Report 2023-24") == "cbe_annual_report_2023_24"
        assert self.agent._make_document_id("Audit Report - 2023") == "audit_report_2023"

    def test_sample_page_indices(self):
        """Sample indices should be evenly distributed."""
        indices = self.agent._sample_page_indices(100, max_samples=10)
        assert len(indices) == 10
        assert indices[0] == 0
        assert indices[-1] < 100

    def test_sample_page_indices_small_doc(self):
        """For small docs, return all pages."""
        indices = self.agent._sample_page_indices(5, max_samples=20)
        assert indices == [0, 1, 2, 3, 4]


class TestConfidenceScoring:
    """Test extraction confidence scoring."""

    def test_avg_confidence_calculation(self):
        """Average confidence should be computed correctly."""
        from src.agents.extractor import ExtractionRouter

        router = ExtractionRouter()

        # Create mock LDUs
        mock_ldus = [MagicMock(confidence=0.9), MagicMock(confidence=0.8), MagicMock(confidence=0.7)]
        avg = router._compute_avg_confidence(mock_ldus)
        assert abs(avg - 0.8) < 0.01

    def test_avg_confidence_empty_list(self):
        """Empty LDU list should return 0.0 confidence."""
        from src.agents.extractor import ExtractionRouter

        router = ExtractionRouter()
        assert router._compute_avg_confidence([]) == 0.0

    def test_strategy_index_lookup(self):
        """Strategy index should map correctly for escalation chain."""
        from src.agents.extractor import ExtractionRouter

        router = ExtractionRouter()
        assert router._strategy_index("fast_text") == 0
        assert router._strategy_index("layout_aware") == 1
        assert router._strategy_index("vision_model") == 2


class TestPydanticModels:
    """Test that all Pydantic models validate correctly."""

    def test_document_profile_creation(self):
        """DocumentProfile should accept valid data."""
        profile = DocumentProfile(
            document_id="test_doc",
            filename="test.pdf",
            file_hash="abc123",
            origin_type=OriginType.NATIVE_DIGITAL,
            layout_complexity=LayoutComplexity.SIMPLE,
            domain_hint=DomainHint.FINANCIAL_REPORT,
            page_count=10,
            recommended_strategy=ExtractionStrategy.STRATEGY_A,
        )
        assert profile.document_id == "test_doc"
        assert profile.page_count == 10

    def test_document_profile_rejects_invalid_page_count(self):
        """Page count must be >= 1."""
        with pytest.raises(Exception):
            DocumentProfile(
                document_id="test",
                filename="test.pdf",
                file_hash="abc",
                origin_type=OriginType.NATIVE_DIGITAL,
                layout_complexity=LayoutComplexity.SIMPLE,
                page_count=0,  # invalid
                recommended_strategy=ExtractionStrategy.STRATEGY_A,
            )

    def test_provenance_chain_creation(self):
        """ProvenanceChain should validate with proper citations."""
        from src.models.provenance import ProvenanceChain, SourceCitation

        chain = ProvenanceChain(
            document_id="test_doc",
            claim="Revenue increased by 15%",
            citations=[
                SourceCitation(
                    page_number=42,
                    text_snippet="Revenue increased by 15% year-over-year.",
                    content_hash="abc123",
                )
            ],
            confidence=0.95,
        )
        assert len(chain.citations) == 1
        assert chain.confidence == 0.95

    def test_ldu_creation(self):
        """LDU should validate with required fields."""
        from src.models.ldu import LDU, LDUType

        ldu = LDU(
            ldu_id="doc_p1_001",
            document_id="test_doc",
            ldu_type=LDUType.PARAGRAPH,
            content="Sample paragraph text.",
            page_number=1,
            content_hash="hash123",
            extraction_strategy="fast_text",
            confidence=0.85,
        )
        assert ldu.ldu_type == "paragraph"

    def test_page_index_creation(self):
        """PageIndex should validate with nested nodes."""
        from src.models.page_index import PageIndex, PageNode

        root = PageNode(
            node_id="root",
            title="Document Title",
            level=0,
            page_start=1,
            page_end=50,
            children=[
                PageNode(
                    node_id="s1",
                    title="Section 1",
                    level=1,
                    page_start=1,
                    page_end=10,
                )
            ],
        )
        index = PageIndex(
            document_id="test_doc",
            root=root,
            total_sections=2,
            max_depth=1,
        )
        assert len(index.root.children) == 1
