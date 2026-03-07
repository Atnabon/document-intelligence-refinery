"""Query Agent — Phase 5 of the Document Intelligence Refinery.

A LangGraph-compatible agent with three tools:
1. pageindex_navigate — Deterministic tree traversal
2. semantic_search    — Vector similarity search (ChromaDB)
3. structured_query   — SQL queries over FactTable (SQLite)

Every answer includes a ProvenanceChain with page numbers, bounding boxes,
and content hashes for audit trail.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.extracted_document import ExtractedDocument
from src.models.ldu import LDU, LDUType
from src.models.page_index import PageIndex
from src.models.provenance import ProvenanceChain
from src.tools.query_tools import (
    AuditMode,
    FactTable,
    VectorStore,
    build_provenance_chain,
    pageindex_navigate,
    semantic_search,
    structured_query,
)

logger = logging.getLogger(__name__)


class QueryAgent:
    """The front-end of the Document Intelligence Refinery.

    Accepts natural language questions about processed documents and returns
    answers with full provenance chains.

    Architecture:
    - Uses PageIndex for section-level navigation (deterministic)
    - Uses VectorStore for semantic similarity search
    - Uses FactTable for structured/numerical queries
    - Combines results with provenance information

    If LangGraph is available, wraps itself as a LangGraph agent with
    proper tool definitions. Otherwise, operates as a standalone agent.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        vector_store: Optional[VectorStore] = None,
        fact_table: Optional[FactTable] = None,
    ):
        self.config = config or {}

        # Initialize data stores
        self.vector_store = vector_store or VectorStore()
        self.fact_table = fact_table or FactTable()
        self.audit = AuditMode(self.vector_store, self.fact_table)

        # Document cache: document_id → (ExtractedDocument, PageIndex)
        self._documents: Dict[str, ExtractedDocument] = {}
        self._indices: Dict[str, PageIndex] = {}
        self._ldu_cache: Dict[str, List[LDU]] = {}

    def register_document(
        self,
        doc: ExtractedDocument,
        page_index: Optional[PageIndex] = None,
    ) -> None:
        """Register a processed document with the query agent.

        Args:
            doc: Fully processed ExtractedDocument.
            page_index: Optional PageIndex (if not embedded in doc).
        """
        doc_id = doc.profile.document_id
        self._documents[doc_id] = doc
        self._ldu_cache[doc_id] = doc.ldus

        # Register PageIndex
        idx = page_index or doc.page_index
        if idx:
            self._indices[doc_id] = idx

        # Ingest into vector store
        self.vector_store.ingest(doc.ldus, doc_id)

        # Ingest into fact table
        self.fact_table.ingest_ldus(doc.ldus, doc_id)

        logger.info(
            "QueryAgent: Registered document %s (%d LDUs)",
            doc_id,
            len(doc.ldus),
        )

    def load_pageindex(self, index_path: Path) -> PageIndex:
        """Load a PageIndex from a JSON file."""
        with open(index_path) as f:
            data = json.load(f)
        index = PageIndex.model_validate(data)
        self._indices[index.document_id] = index
        return index

    def query(
        self,
        question: str,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a question about a registered document.

        Strategy:
        1. If a PageIndex exists, navigate to relevant sections first.
        2. Perform semantic search filtered to relevant sections.
        3. Check FactTable for numerical answers.
        4. Synthesize answer with provenance.

        Args:
            question: Natural language question.
            document_id: Document to query (if None, searches all).

        Returns:
            Dict with: answer, provenance_chain, sources, method.
        """
        logger.info("QueryAgent: Question='%s', doc=%s", question[:80], document_id)

        results = {
            "question": question,
            "document_id": document_id,
            "answer": "",
            "provenance_chain": None,
            "sources": [],
            "method": [],
        }

        # Determine target documents
        target_docs = (
            [document_id] if document_id else list(self._documents.keys())
        )

        if not target_docs:
            results["answer"] = "No documents registered. Please process a document first."
            return results

        all_source_ldus: List[LDU] = []
        actual_doc_id = target_docs[0] if target_docs else ""
        doc_name = ""

        for doc_id in target_docs:
            doc = self._documents.get(doc_id)
            if doc:
                doc_name = doc.profile.filename
                actual_doc_id = doc_id

            # Step 1: PageIndex navigation
            index = self._indices.get(doc_id)
            relevant_ldu_ids = set()

            if index:
                nav_results = pageindex_navigate(index, question, top_k=3)
                results["method"].append("pageindex_navigate")

                for nav in nav_results:
                    relevant_ldu_ids.update(nav.get("ldu_ids", []))
                    results["sources"].append({
                        "tool": "pageindex_navigate",
                        "section": nav.get("title", ""),
                        "pages": f"{nav.get('page_start', '?')}–{nav.get('page_end', '?')}",
                        "relevance": nav.get("relevance_score", 0),
                    })

            # Step 2: Semantic search
            search_results = semantic_search(
                self.vector_store, question, doc_id, top_k=5
            )
            results["method"].append("semantic_search")

            for sr in search_results:
                results["sources"].append({
                    "tool": "semantic_search",
                    "ldu_id": sr.get("ldu_id", ""),
                    "page": sr.get("metadata", {}).get("page_number", "?"),
                    "distance": sr.get("distance", 0),
                    "snippet": sr.get("content", "")[:200],
                })

            # Step 3: Structured query for numerical questions
            numerical_indicators = [
                "how much", "total", "revenue", "cost", "amount",
                "price", "rate", "percentage", "number", "count",
                "value", "budget", "income", "expenditure", "tax",
            ]
            if any(ind in question.lower() for ind in numerical_indicators):
                fact_results = structured_query(
                    self.fact_table, question, doc_id
                )
                results["method"].append("structured_query")
                for fr in fact_results:
                    if "error" not in fr:
                        results["sources"].append({
                            "tool": "structured_query",
                            "fact": fr,
                        })

            # Collect source LDUs for provenance
            ldus = self._ldu_cache.get(doc_id, [])

            # Prefer LDUs from PageIndex navigation
            if relevant_ldu_ids:
                for ldu in ldus:
                    if ldu.ldu_id in relevant_ldu_ids:
                        all_source_ldus.append(ldu)

            # Add LDUs from semantic search
            for sr in search_results:
                matching = [l for l in ldus if l.ldu_id == sr.get("ldu_id", "")]
                all_source_ldus.extend(matching)

        # Deduplicate source LDUs
        seen_ids = set()
        unique_ldus = []
        for ldu in all_source_ldus:
            if ldu.ldu_id not in seen_ids:
                seen_ids.add(ldu.ldu_id)
                unique_ldus.append(ldu)

        # Synthesize answer
        answer = self._synthesize_answer(question, unique_ldus[:10], results["sources"])
        results["answer"] = answer

        # Build provenance chain
        if unique_ldus:
            avg_conf = sum(l.confidence for l in unique_ldus[:5]) / min(len(unique_ldus), 5)
            provenance = build_provenance_chain(
                document_id=actual_doc_id,
                document_name=doc_name,
                claim=answer,
                source_ldus=unique_ldus[:5],
                confidence=avg_conf,
            )
            results["provenance_chain"] = provenance.model_dump()

        return results

    def verify_claim(
        self,
        claim: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """Audit Mode: Verify a claim against the document corpus.

        Args:
            claim: The claim to verify.
            document_id: Document to check against.

        Returns:
            Dict with verification result and provenance.
        """
        doc = self._documents.get(document_id)
        doc_name = doc.profile.filename if doc else document_id

        chain = self.audit.verify_claim(claim, document_id, doc_name)

        return {
            "claim": claim,
            "verification_status": chain.verification_status,
            "confidence": chain.confidence,
            "provenance_chain": chain.model_dump(),
        }

    def _synthesize_answer(
        self,
        question: str,
        source_ldus: List[LDU],
        sources: List[Dict],
    ) -> str:
        """Synthesize an answer from source LDUs.

        Uses LLM if available, otherwise returns extracted content directly.
        """
        if not source_ldus:
            return "No relevant information found in the processed documents."

        # Collect context
        context_parts = []
        for ldu in source_ldus[:5]:
            prefix = ""
            if ldu.section_heading:
                prefix = f"[Section: {ldu.section_heading}] "
            page_ref = f"(p.{ldu.page_number})"
            context_parts.append(f"{prefix}{ldu.content[:500]} {page_ref}")

        context = "\n\n".join(context_parts)

        # Try LLM synthesis
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                return self._llm_answer(question, context)
            except Exception as e:
                logger.warning("LLM synthesis failed: %s. Using direct extraction.", e)

        # Fallback: direct extraction
        # Build answer from the most relevant content
        answer_parts = []
        for ldu in source_ldus[:3]:
            snippet = ldu.content.strip()[:300]
            page_ref = f"(Page {ldu.page_number})"
            answer_parts.append(f"{snippet} {page_ref}")

        # Add fact table results if available
        for src in sources:
            if src.get("tool") == "structured_query" and "fact" in src:
                fact = src["fact"]
                if isinstance(fact, dict) and "key" in fact:
                    answer_parts.append(
                        f"Fact: {fact.get('key', '')} = {fact.get('value', '')} (p.{fact.get('page_number', '?')})"
                    )

        return "Based on the document:\n\n" + "\n\n".join(answer_parts)

    def _llm_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM."""
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document analysis assistant. Answer the question based ONLY on "
                        "the provided context. Include page references. If the answer is not in the "
                        "context, say so. Be precise and cite specific numbers."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=500,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def get_qa_example(
        self,
        question: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """Generate a complete Q&A example with full ProvenanceChain.

        Used for generating the 12 required Q&A examples (3 per class).
        """
        result = self.query(question, document_id)

        return {
            "document_id": document_id,
            "document_class": self._get_doc_class(document_id),
            "question": question,
            "answer": result["answer"],
            "methods_used": result["method"],
            "source_count": len(result["sources"]),
            "provenance_chain": result.get("provenance_chain"),
        }

    def _get_doc_class(self, document_id: str) -> str:
        """Determine document class from profile."""
        doc = self._documents.get(document_id)
        if not doc:
            return "unknown"
        domain = doc.profile.domain_hint
        if domain == "financial_report":
            origin = doc.profile.origin_type
            if origin in ("scanned_image",):
                return "B"
            return "A"
        elif domain == "legal_audit":
            return "B"
        elif domain == "technical_assessment":
            return "C"
        elif domain == "structured_data":
            return "D"
        return "unknown"
