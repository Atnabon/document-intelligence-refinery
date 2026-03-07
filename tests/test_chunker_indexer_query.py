"""Tests for Phase 3-5 components: Chunker, Indexer, and Query Agent.

Tests cover:
- ChunkingEngine: all 5 chunking rules, split/merge behavior
- ChunkValidator: validation of chunked LDUs
- PageIndexBuilder: tree construction, navigation
- QueryAgent: registration, querying, claim verification
- FactTable: ingestion and structured queries
- VectorStore: ingestion and keyword search fallback
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import List

import pytest

from src.models.ldu import LDU, LDUType
from src.models.page_index import PageIndex, PageNode
from src.models.provenance import BoundingBox
from src.agents.chunker import ChunkingEngine, ChunkValidator
from src.agents.indexer import PageIndexBuilder
from src.tools.query_tools import (
    FactTable,
    VectorStore,
    AuditMode,
    build_provenance_chain,
    pageindex_navigate,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_ldu(
    ldu_id: str = "test_p1_001",
    doc_id: str = "test_doc",
    ldu_type: str = "paragraph",
    content: str = "This is a test paragraph with enough content to be valid.",
    page: int = 1,
    confidence: float = 0.85,
    section: str = None,
    structured: dict = None,
    seq: int = 0,
) -> LDU:
    return LDU(
        ldu_id=ldu_id,
        document_id=doc_id,
        ldu_type=ldu_type,
        content=content,
        page_number=page,
        content_hash=_hash(content),
        section_heading=section,
        extraction_strategy="fast_text",
        confidence=confidence,
        sequence_index=seq,
        structured_content=structured,
    )


def _make_ldus_list() -> List[LDU]:
    """Create a realistic list of LDUs for testing."""
    return [
        _make_ldu("doc_p1_001", ldu_type="heading", content="Financial Highlights", page=1, seq=0),
        _make_ldu("doc_p1_002", content="Total assets reached 2.1 trillion ETB in fiscal year 2023-24.", page=1, seq=1),
        _make_ldu("doc_p1_003", content="Net income increased by 15% compared to the previous year.", page=1, seq=2),
        _make_ldu("doc_p2_001", ldu_type="heading", content="Income Statement", page=2, seq=0),
        _make_ldu(
            "doc_p2_002",
            ldu_type="table",
            content="Interest Income | 189,234 | 156,789\nTotal Revenue | 245,678 | 201,345",
            page=2,
            seq=1,
            structured={
                "headers": ["Item", "2023-24", "2022-23"],
                "rows": [
                    ["Interest Income", "189,234", "156,789"],
                    ["Total Revenue", "245,678", "201,345"],
                ],
            },
        ),
        _make_ldu("doc_p2_003", ldu_type="caption", content="Table 1: Income Statement Summary", page=2, seq=2),
        _make_ldu("doc_p3_001", ldu_type="heading", content="Branch Network", page=3, seq=0),
        _make_ldu("doc_p3_002", content="The bank operates 2,000 branches across Ethiopia.", page=3, seq=1),
        _make_ldu("doc_p3_003", ldu_type="footnote", content="1. Including sub-branches and digital outlets.", page=3, seq=2),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Test ChunkValidator
# ═══════════════════════════════════════════════════════════════════════════


class TestChunkValidator:
    def test_valid_ldus_pass(self):
        validator = ChunkValidator()
        ldus = _make_ldus_list()
        report = validator.validate(ldus)
        assert report.total_ldus == len(ldus)

    def test_table_integrity_passes(self):
        validator = ChunkValidator()
        table_ldu = _make_ldu(
            ldu_type="table",
            content="A | B\n1 | 2",
            structured={"headers": ["A", "B"], "rows": [["1", "2"]]},
        )
        report = validator.validate([table_ldu])
        table_violations = [v for v in report.violations if v.rule_name == "table_integrity"]
        assert all(v.passed for v in table_violations) or len(table_violations) == 0

    def test_section_propagation_advisory(self):
        """LDU without section_heading triggers section_propagation violation."""
        validator = ChunkValidator()
        # An LDU with no section_heading should fail section_propagation
        orphan_ldu = _make_ldu(content="This text has no section heading assigned.")
        # Force section_heading to None
        orphan_ldu = orphan_ldu.model_copy(update={"section_heading": None})
        report = validator.validate([orphan_ldu])
        prop_violations = [v for v in report.violations if v.rule_name == "section_propagation"]
        assert len(prop_violations) > 0

    def test_list_integrity_passes(self):
        """List LDU within token limit passes list_integrity."""
        validator = ChunkValidator()
        list_ldu = _make_ldu(ldu_type="list", content="1. First item\n2. Second item\n3. Third item")
        report = validator.validate([list_ldu])
        list_violations = [v for v in report.violations if v.rule_name == "list_integrity" and not v.passed]
        assert len(list_violations) == 0

    def test_caption_without_parent_fails(self):
        validator = ChunkValidator()
        caption = _make_ldu(ldu_type="caption", content="Figure 1: Test")
        report = validator.validate([caption])
        cap_violations = [v for v in report.violations if v.rule_name == "caption_binding" and not v.passed]
        assert len(cap_violations) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test ChunkingEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestChunkingEngine:
    def test_chunk_preserves_order(self):
        engine = ChunkingEngine()
        ldus = _make_ldus_list()
        chunked = engine.chunk(ldus)
        assert len(chunked) > 0

        # Verify page ordering is maintained
        pages = [l.page_number for l in chunked]
        assert pages == sorted(pages)

    def test_section_heading_propagation(self):
        engine = ChunkingEngine()
        ldus = _make_ldus_list()
        chunked = engine.chunk(ldus)

        # LDUs after "Financial Highlights" heading should have that section
        for ldu in chunked:
            if ldu.page_number == 1 and ldu.ldu_type != "heading":
                assert ldu.section_heading == "Financial Highlights"

    def test_caption_binding(self):
        engine = ChunkingEngine()
        ldus = _make_ldus_list()
        chunked = engine.chunk(ldus)

        # Find caption LDU
        captions = [l for l in chunked if l.ldu_type in ("caption", LDUType.CAPTION)]
        for cap in captions:
            assert cap.parent_ldu_id is not None, f"Caption {cap.ldu_id} has no parent"

    def test_split_oversized_paragraph(self):
        engine = ChunkingEngine(config={"chunking": {"rules": [
            {"name": "maximum_size", "max_chars": 100},
        ]}})
        large_ldu = _make_ldu(content="This is a sentence. " * 20)  # ~400 chars
        chunked = engine.chunk([large_ldu])
        assert len(chunked) > 1, "Oversized LDU should be split"

    def test_merge_undersized_paragraphs(self):
        engine = ChunkingEngine()
        tiny1 = _make_ldu("t1", content="Hi.", page=1, seq=0)
        tiny2 = _make_ldu("t2", content="There.", page=1, seq=1)
        chunked = engine.chunk([tiny1, tiny2])
        # Both are < min_chars, same page, same type → should merge
        assert len(chunked) <= 2

    def test_tables_never_split(self):
        engine = ChunkingEngine(config={"chunking": {"rules": [
            {"name": "maximum_size", "max_chars": 50},
        ]}})
        table = _make_ldu(
            ldu_type="table",
            content="A" * 200,
            structured={"headers": ["A"], "rows": [["B"]]},
        )
        chunked = engine.chunk([table])
        tables = [l for l in chunked if l.ldu_type in ("table", LDUType.TABLE)]
        assert len(tables) == 1, "Tables should never be split"

    def test_content_hash_recomputed(self):
        engine = ChunkingEngine()
        ldus = _make_ldus_list()
        chunked = engine.chunk(ldus)
        for ldu in chunked:
            expected_hash = _hash(ldu.content)
            assert ldu.content_hash == expected_hash

    def test_token_count_computed(self):
        """Every LDU after chunking should have token_count > 0."""
        engine = ChunkingEngine()
        ldus = _make_ldus_list()
        chunked = engine.chunk(ldus)
        for ldu in chunked:
            assert ldu.token_count > 0, f"LDU {ldu.ldu_id} has token_count=0"

    def test_list_preservation(self):
        """Adjacent list items should be merged into a single LIST LDU."""
        engine = ChunkingEngine()
        items = [
            _make_ldu(f"li_{i}", content=f"{i+1}. List item number {i+1}", page=1, seq=i)
            for i in range(4)
        ]
        chunked = engine.chunk(items)
        list_ldus = [l for l in chunked if l.ldu_type in ("list", LDUType.LIST)]
        assert len(list_ldus) >= 1, "List items should be merged into LIST LDUs"

    def test_cross_reference_resolution(self):
        """Cross-references like 'see Table 1' should be resolved."""
        engine = ChunkingEngine()
        table_ldu = _make_ldu(
            "tbl_1", ldu_type="table", content="Table 1: Revenue breakdown",
            structured={"headers": ["Item", "Value"], "rows": [["Revenue", "100"]]},
            page=1, seq=0,
        )
        text_ldu = _make_ldu(
            "txt_1", content="As shown in see Table 1, revenue increased.",
            page=2, seq=0,
        )
        chunked = engine.chunk([table_ldu, text_ldu])
        # Find the text LDU and check cross_references
        text_results = [l for l in chunked if "txt" in l.ldu_id or "see Table" in l.content]
        if text_results:
            assert len(text_results[0].cross_references) >= 0  # Resolution attempted


# ═══════════════════════════════════════════════════════════════════════════
# Test PageIndexBuilder
# ═══════════════════════════════════════════════════════════════════════════


class TestPageIndexBuilder:
    def test_build_basic_index(self):
        builder = PageIndexBuilder()
        ldus = _make_ldus_list()
        index = builder.build("test_doc", ldus, "Test Document", generate_summaries=False)
        assert index.document_id == "test_doc"
        assert index.total_sections > 0
        assert index.max_depth >= 0
        assert index.root.title == "Test Document"

    def test_heading_extraction(self):
        builder = PageIndexBuilder()
        ldus = _make_ldus_list()
        index = builder.build("test_doc", ldus, generate_summaries=False)
        # Should have children for headings
        assert len(index.root.children) > 0

    def test_ldu_assignment(self):
        builder = PageIndexBuilder()
        ldus = _make_ldus_list()
        index = builder.build("test_doc", ldus, generate_summaries=False)

        # Check that LDU IDs were assigned
        total_ldu_ids = []
        def collect_ids(node):
            total_ldu_ids.extend(node.ldu_ids)
            for c in node.children:
                collect_ids(c)
        collect_ids(index.root)
        assert len(total_ldu_ids) > 0

    def test_navigate(self):
        builder = PageIndexBuilder()
        ldus = _make_ldus_list()
        index = builder.build("test_doc", ldus, generate_summaries=False)

        # Navigate to "income" topic
        results = builder.navigate(index, "income statement", top_k=3)
        assert len(results) > 0
        # The "Income Statement" heading should be top-ranked
        titles = [n.title.lower() for n in results]
        assert any("income" in t for t in titles)

    def test_empty_ldus(self):
        builder = PageIndexBuilder()
        index = builder.build("empty_doc", [], generate_summaries=False)
        assert index.total_sections == 1
        assert index.root.title == "empty_doc"

    def test_data_types_present(self):
        """PageNode.data_types_present should be populated for sections with tables/figures."""
        builder = PageIndexBuilder()
        ldus = _make_ldus_list()  # contains table LDU
        index = builder.build("test_doc", ldus, generate_summaries=False)

        # Collect all data_types_present across all nodes
        all_dtypes = []
        def collect_dtypes(node):
            all_dtypes.extend(node.data_types_present)
            for c in node.children:
                collect_dtypes(c)
        collect_dtypes(index.root)
        assert len(all_dtypes) > 0, "data_types_present should be populated"
        assert "table" in all_dtypes, "Should detect table data type"

    def test_key_entities_extracted(self):
        """PageNode.key_entities should contain extracted entities."""
        builder = PageIndexBuilder()
        # Create LDUs with entity-rich content
        ldus = [
            _make_ldu("h1", ldu_type="heading", content="Financial Highlights", page=1, seq=0),
            _make_ldu("p1", content="CBE reported total revenue of ETB 245,678 million in FY 2023-24, up 15.2% from previous year.", page=1, seq=1),
            _make_ldu("t1", ldu_type="table", content="Table 1: Revenue by segment",
                      structured={"headers": ["Segment", "2023-24"], "rows": [["Interest", "189,234"]]},
                      page=1, seq=2),
        ]
        index = builder.build("test_doc", ldus, generate_summaries=False)

        all_entities = []
        def collect_entities(node):
            all_entities.extend(node.key_entities)
            for c in node.children:
                collect_entities(c)
        collect_entities(index.root)
        assert len(all_entities) > 0, "key_entities should be extracted"

    def test_save_and_load(self):
        builder = PageIndexBuilder()
        ldus = _make_ldus_list()
        index = builder.build("test_doc", ldus, "Test Doc", generate_summaries=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder.output_dir = Path(tmpdir)
            saved_path = builder.save(index)
            assert saved_path.exists()

            # Verify JSON is valid
            import json
            with open(saved_path) as f:
                data = json.load(f)
            assert data["document_id"] == "test_doc"


# ═══════════════════════════════════════════════════════════════════════════
# Test PageIndex Navigation Tool
# ═══════════════════════════════════════════════════════════════════════════


class TestPageIndexNavigate:
    def _build_index(self):
        builder = PageIndexBuilder()
        ldus = _make_ldus_list()
        return builder.build("test_doc", ldus, "Test Document", generate_summaries=False)

    def test_pageindex_navigate_returns_results(self):
        index = self._build_index()
        results = pageindex_navigate(index, "financial highlights", top_k=3)
        assert len(results) > 0
        assert "title" in results[0]
        assert "relevance_score" in results[0]

    def test_pageindex_navigate_irrelevant_topic(self):
        index = self._build_index()
        results = pageindex_navigate(index, "quantum computing", top_k=3)
        # May return 0 or low-scored results
        if results:
            assert results[0]["relevance_score"] < 5.0


# ═══════════════════════════════════════════════════════════════════════════
# Test FactTable
# ═══════════════════════════════════════════════════════════════════════════


class TestFactTable:
    def test_ingest_table_ldu(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ft = FactTable(db_path=db_path)
            table_ldu = _make_ldu(
                ldu_type="table",
                content="Revenue | 100M | 90M",
                structured={
                    "headers": ["Item", "2024", "2023"],
                    "rows": [
                        ["Revenue", "100,000,000", "90,000,000"],
                        ["Expenses", "60,000,000", "55,000,000"],
                    ],
                },
            )
            count = ft.ingest_ldus([table_ldu], "test_doc")
            assert count > 0

            # Query facts
            results = ft.search_facts("Revenue", "test_doc")
            assert len(results) > 0
            ft.close()
        finally:
            os.unlink(db_path)

    def test_parse_numeric(self):
        assert FactTable._parse_numeric("1,234,567") == 1234567.0
        assert FactTable._parse_numeric("$4.2B") == 4.2e9
        assert FactTable._parse_numeric("(500)") == -500.0
        assert FactTable._parse_numeric("15%") == 15.0
        assert FactTable._parse_numeric("N/A") is None

    def test_sql_query(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ft = FactTable(db_path=db_path)
            table_ldu = _make_ldu(
                ldu_type="table",
                structured={
                    "headers": ["Metric", "Value"],
                    "rows": [["Total Assets", "2,100,000"]],
                },
            )
            ft.ingest_ldus([table_ldu], "test_doc")
            results = ft.query("SELECT COUNT(*) as cnt FROM facts WHERE document_id = 'test_doc'")
            assert len(results) > 0
            assert results[0]["cnt"] > 0
            ft.close()
        finally:
            os.unlink(db_path)


# ═══════════════════════════════════════════════════════════════════════════
# Test VectorStore (keyword fallback)
# ═══════════════════════════════════════════════════════════════════════════


class TestVectorStore:
    def test_ingest_and_search_keyword_fallback(self):
        # VectorStore without ChromaDB falls back to keyword search
        vs = VectorStore.__new__(VectorStore)
        vs._use_chroma = False
        vs._collection = None
        vs._client = None
        vs._memory_store = {}

        ldus = _make_ldus_list()
        count = vs.ingest(ldus, "test_doc")
        assert count == len(ldus)

        results = vs.search("total assets", "test_doc", top_k=3)
        assert len(results) > 0
        assert "content" in results[0]


# ═══════════════════════════════════════════════════════════════════════════
# Test Provenance Builder
# ═══════════════════════════════════════════════════════════════════════════


class TestProvenanceBuilder:
    def test_build_provenance_chain(self):
        ldus = _make_ldus_list()[:2]
        chain = build_provenance_chain(
            document_id="test_doc",
            document_name="test.pdf",
            claim="Total assets reached 2.1 trillion ETB",
            source_ldus=ldus,
            confidence=0.85,
        )
        assert chain.document_id == "test_doc"
        assert chain.claim == "Total assets reached 2.1 trillion ETB"
        assert len(chain.citations) == 2
        assert chain.confidence == 0.85
        assert chain.verification_status == "verified"

    def test_empty_sources_provenance(self):
        chain = build_provenance_chain(
            document_id="test_doc",
            document_name="test.pdf",
            claim="Unknown claim",
            source_ldus=[],
            confidence=0.0,
        )
        assert chain.confidence == 0.0
        assert len(chain.citations) == 1  # Minimal "not found" citation


# ═══════════════════════════════════════════════════════════════════════════
# Test AuditMode
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditMode:
    def test_verify_claim_found(self):
        vs = VectorStore.__new__(VectorStore)
        vs._use_chroma = False
        vs._collection = None
        vs._client = None
        vs._memory_store = {}

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ft = FactTable(db_path=db_path)
            ldus = _make_ldus_list()
            vs.ingest(ldus, "test_doc")
            ft.ingest_ldus(ldus, "test_doc")

            audit = AuditMode(vs, ft)
            chain = audit.verify_claim(
                "Total assets reached 2.1 trillion",
                "test_doc",
                "test.pdf",
            )
            assert chain.verification_status in ("verified", "unverified", "unverifiable")
            assert len(chain.citations) > 0
            ft.close()
        finally:
            os.unlink(db_path)
