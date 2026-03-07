"""Query tools for the Document Intelligence Refinery Query Agent.

Three tools implementing the retrieval layer:
1. pageindex_navigate — Deterministic tree traversal over PageIndex
2. semantic_search    — Vector similarity search over LDU embeddings (ChromaDB)
3. structured_query   — SQL queries over the FactTable (SQLite)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.ldu import LDU, LDUType
from src.models.page_index import PageIndex, PageNode
from src.models.provenance import BoundingBox, ProvenanceChain, SourceCitation

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 1: PageIndex Navigate
# ═══════════════════════════════════════════════════════════════════════════


def pageindex_navigate(
    index: PageIndex,
    topic: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Navigate the PageIndex tree to find sections relevant to a topic.

    This is a deterministic, non-probabilistic traversal that uses keyword
    matching against section titles, summaries, and metadata.

    Args:
        index: PageIndex tree for the document.
        topic: Natural language topic to search for.
        top_k: Number of top sections to return.

    Returns:
        List of dicts with section info (title, pages, summary, ldu_ids).
    """
    topic_lower = topic.lower()
    topic_words = set(re.findall(r"\w+", topic_lower))

    scored: List[Tuple[float, PageNode]] = []
    _score_tree(index.root, topic_lower, topic_words, scored)

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, node in scored[:top_k]:
        results.append({
            "node_id": node.node_id,
            "title": node.title,
            "level": node.level,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "summary": node.summary,
            "ldu_ids": node.ldu_ids,
            "relevance_score": round(score, 3),
            "data_types": node.metadata.get("data_types", ""),
        })

    return results


def _score_tree(
    node: PageNode,
    topic_lower: str,
    topic_words: set,
    results: List[Tuple[float, PageNode]],
) -> None:
    """Recursively score nodes for relevance."""
    score = 0.0
    title_lower = node.title.lower()
    title_words = set(re.findall(r"\w+", title_lower))

    # Word overlap scoring
    word_overlap = len(topic_words & title_words)
    score += word_overlap * 3.0

    # Exact substring match
    if topic_lower in title_lower:
        score += 5.0

    # Summary scoring
    if node.summary:
        summary_lower = node.summary.lower()
        summary_words = set(re.findall(r"\w+", summary_lower))
        score += len(topic_words & summary_words) * 1.0
        if topic_lower in summary_lower:
            score += 3.0

    # Metadata scoring
    for key, value in node.metadata.items():
        if any(w in value.lower() for w in topic_words):
            score += 1.0

    # Boost leaf nodes (more specific sections)
    if not node.children:
        score *= 1.2

    if score > 0:
        results.append((score, node))

    for child in node.children:
        _score_tree(child, topic_lower, topic_words, results)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 2: Semantic Search (ChromaDB)
# ═══════════════════════════════════════════════════════════════════════════


class VectorStore:
    """ChromaDB-backed vector store for LDU semantic search.

    Ingests LDUs with their metadata and provides similarity search.
    Falls back to keyword search if ChromaDB is not available.
    """

    def __init__(self, collection_name: str = "refinery_ldus", persist_dir: str = ".refinery/vectorstore"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._collection = None
        self._client = None
        self._use_chroma = False

        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
            logger.info("VectorStore: ChromaDB initialized at %s", persist_dir)
        except ImportError:
            logger.warning(
                "ChromaDB not installed. Falling back to keyword search. "
                "Install with: pip install chromadb"
            )
        except Exception as e:
            logger.warning("ChromaDB init failed: %s. Using keyword fallback.", e)

    def ingest(self, ldus: List[LDU], document_id: str) -> int:
        """Ingest LDUs into the vector store.

        Args:
            ldus: List of LDUs to ingest.
            document_id: Document identifier for metadata.

        Returns:
            Number of LDUs ingested.
        """
        if not ldus:
            return 0

        if self._use_chroma and self._collection is not None:
            return self._ingest_chroma(ldus, document_id)
        else:
            # Store in memory for keyword fallback
            if not hasattr(self, "_memory_store"):
                self._memory_store: Dict[str, LDU] = {}
            for ldu in ldus:
                self._memory_store[ldu.ldu_id] = ldu
            return len(ldus)

    def _ingest_chroma(self, ldus: List[LDU], document_id: str) -> int:
        """Ingest LDUs into ChromaDB."""
        ids = []
        documents = []
        metadatas = []

        for ldu in ldus:
            ids.append(ldu.ldu_id)
            documents.append(ldu.content[:8000])  # ChromaDB text limit
            ldu_type = ldu.ldu_type if isinstance(ldu.ldu_type, str) else ldu.ldu_type.value
            metadatas.append({
                "document_id": document_id,
                "chunk_type": ldu_type,
                "ldu_type": ldu_type,
                "page_refs": ldu.page_number,
                "page_number": ldu.page_number,
                "parent_section": ldu.section_heading or "",
                "section_heading": ldu.section_heading or "",
                "confidence": ldu.confidence,
                "content_hash": ldu.content_hash,
                "token_count": ldu.token_count,
            })

        # Batch upsert (ChromaDB max batch = 5461)
        batch_size = 5000
        ingested = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i: i + batch_size]
            batch_docs = documents[i: i + batch_size]
            batch_meta = metadatas[i: i + batch_size]
            self._collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            ingested += len(batch_ids)

        logger.info("VectorStore: Ingested %d LDUs for %s", ingested, document_id)
        return ingested

    def search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for LDUs similar to the query.

        Args:
            query: Natural language query.
            document_id: Optional filter by document.
            top_k: Number of results.

        Returns:
            List of dicts with ldu_id, content, metadata, and distance.
        """
        if self._use_chroma and self._collection is not None:
            return self._search_chroma(query, document_id, top_k)
        else:
            return self._search_keyword(query, document_id, top_k)

    def _search_chroma(
        self,
        query: str,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """ChromaDB similarity search."""
        where_filter = None
        if document_id:
            where_filter = {"document_id": document_id}

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        output = []
        if results and results["ids"]:
            for i, ldu_id in enumerate(results["ids"][0]):
                output.append({
                    "ldu_id": ldu_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })

        return output

    def _search_keyword(
        self,
        query: str,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Keyword-based fallback search."""
        if not hasattr(self, "_memory_store"):
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []

        for ldu_id, ldu in self._memory_store.items():
            if document_id and ldu.document_id != document_id:
                continue
            content_words = set(re.findall(r"\w+", ldu.content.lower()))
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap, ldu))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "ldu_id": ldu.ldu_id,
                "content": ldu.content[:500],
                "metadata": {
                    "document_id": ldu.document_id,
                    "chunk_type": ldu.ldu_type if isinstance(ldu.ldu_type, str) else ldu.ldu_type.value,
                    "ldu_type": ldu.ldu_type if isinstance(ldu.ldu_type, str) else ldu.ldu_type.value,
                    "page_refs": ldu.page_number,
                    "page_number": ldu.page_number,
                    "parent_section": ldu.section_heading or "",
                    "section_heading": ldu.section_heading or "",
                    "confidence": ldu.confidence,
                    "content_hash": ldu.content_hash,
                    "token_count": ldu.token_count,
                },
                "distance": 1.0 / (score + 1),
            }
            for score, ldu in scored[:top_k]
        ]


def semantic_search(
    vector_store: VectorStore,
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Perform semantic search over ingested LDUs.

    Args:
        vector_store: Initialized VectorStore instance.
        query: Natural language query.
        document_id: Optional document filter.
        top_k: Number of results.

    Returns:
        List of matching LDU results with content and metadata.
    """
    return vector_store.search(query, document_id, top_k)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 3: Structured Query (SQLite FactTable)
# ═══════════════════════════════════════════════════════════════════════════


class FactTable:
    """SQLite-backed fact table for structured numerical queries.

    Extracts key-value facts from table LDUs and stores them in a queryable
    SQLite database. Supports SQL queries for precise numerical retrieval.
    """

    def __init__(self, db_path: str = ".refinery/facts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        """Create the fact tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                numeric_value REAL,
                unit TEXT,
                page_number INTEGER,
                section_heading TEXT,
                source_ldu_id TEXT,
                content_hash TEXT
            );

            CREATE TABLE IF NOT EXISTS table_data (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                table_ldu_id TEXT NOT NULL,
                row_index INTEGER,
                col_header TEXT,
                cell_value TEXT,
                numeric_value REAL,
                page_number INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(document_id);
            CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
            CREATE INDEX IF NOT EXISTS idx_table_doc ON table_data(document_id);
            CREATE INDEX IF NOT EXISTS idx_table_header ON table_data(col_header);
        """)
        self.conn.commit()

    def ingest_ldus(self, ldus: List[LDU], document_id: str) -> int:
        """Extract facts from LDUs and insert into the database.

        Processes:
        - Table LDUs: each cell → table_data row; header-value pairs → facts
        - Key-Value LDUs: directly → facts

        Returns:
            Number of facts ingested.
        """
        fact_count = 0

        for ldu in ldus:
            if ldu.ldu_type in (LDUType.TABLE, "table"):
                fact_count += self._ingest_table(ldu, document_id)
            elif ldu.ldu_type in (LDUType.KEY_VALUE, "key_value"):
                fact_count += self._ingest_key_value(ldu, document_id)

        self.conn.commit()
        logger.info("FactTable: Ingested %d facts for %s", fact_count, document_id)
        return fact_count

    def _ingest_table(self, ldu: LDU, document_id: str) -> int:
        """Extract facts from a table LDU."""
        if not ldu.structured_content:
            return 0

        headers = ldu.structured_content.get("headers", [])
        rows = ldu.structured_content.get("rows", [])
        count = 0

        for row_idx, row in enumerate(rows):
            for col_idx, cell_value in enumerate(row):
                if col_idx >= len(headers):
                    col_header = f"col_{col_idx}"
                else:
                    col_header = str(headers[col_idx]) if headers[col_idx] else f"col_{col_idx}"

                numeric = self._parse_numeric(str(cell_value) if cell_value else "")

                try:
                    self.conn.execute(
                        """INSERT OR REPLACE INTO table_data
                        (document_id, table_ldu_id, row_index, col_header,
                         cell_value, numeric_value, page_number)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            document_id,
                            ldu.ldu_id,
                            row_idx,
                            col_header,
                            str(cell_value) if cell_value else "",
                            numeric,
                            ldu.page_number,
                        ),
                    )
                    count += 1
                except Exception as e:
                    logger.debug("FactTable insert error: %s", e)

            # Also create key-value facts for first column = key, other cols = values
            if len(row) >= 2 and headers:
                key = str(row[0]).strip() if row[0] else ""
                if key and len(key) < 200:
                    for col_idx in range(1, min(len(row), len(headers))):
                        value = str(row[col_idx]) if row[col_idx] else ""
                        if value.strip():
                            fact_id = hashlib.sha256(
                                f"{document_id}_{key}_{headers[col_idx]}_{row_idx}".encode()
                            ).hexdigest()[:16]
                            numeric = self._parse_numeric(value)
                            try:
                                self.conn.execute(
                                    """INSERT OR REPLACE INTO facts
                                    (fact_id, document_id, key, value, numeric_value,
                                     unit, page_number, section_heading, source_ldu_id, content_hash)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        fact_id,
                                        document_id,
                                        f"{key} ({headers[col_idx]})" if headers[col_idx] else key,
                                        value,
                                        numeric,
                                        "",
                                        ldu.page_number,
                                        ldu.section_heading or "",
                                        ldu.ldu_id,
                                        ldu.content_hash,
                                    ),
                                )
                                count += 1
                            except Exception as e:
                                logger.debug("Fact insert error: %s", e)

        return count

    def _ingest_key_value(self, ldu: LDU, document_id: str) -> int:
        """Extract facts from a key-value LDU."""
        # Parse "Key: Value" patterns from content
        kv_patterns = re.findall(
            r"([A-Za-z][A-Za-z\s]{2,50}):\s*(.+?)(?:\n|$)", ldu.content
        )
        count = 0
        for key, value in kv_patterns:
            key = key.strip()
            value = value.strip()
            if not value:
                continue

            fact_id = hashlib.sha256(
                f"{document_id}_{key}_{ldu.page_number}".encode()
            ).hexdigest()[:16]
            numeric = self._parse_numeric(value)

            try:
                self.conn.execute(
                    """INSERT OR REPLACE INTO facts
                    (fact_id, document_id, key, value, numeric_value,
                     unit, page_number, section_heading, source_ldu_id, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        fact_id,
                        document_id,
                        key,
                        value,
                        numeric,
                        "",
                        ldu.page_number,
                        ldu.section_heading or "",
                        ldu.ldu_id,
                        ldu.content_hash,
                    ),
                )
                count += 1
            except Exception as e:
                logger.debug("KV fact insert error: %s", e)

        return count

    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SQL query against the fact tables.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            List of result dicts.
        """
        try:
            cursor = self.conn.execute(sql, params)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error("FactTable query error: %s", e)
            return [{"error": str(e)}]

    def search_facts(
        self,
        keyword: str,
        document_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search facts by keyword in key or value fields.

        Args:
            keyword: Search term.
            document_id: Optional document filter.
            limit: Max results.

        Returns:
            List of matching fact dicts.
        """
        if document_id:
            return self.query(
                """SELECT * FROM facts
                WHERE document_id = ?
                AND (key LIKE ? OR value LIKE ?)
                LIMIT ?""",
                (document_id, f"%{keyword}%", f"%{keyword}%", limit),
            )
        return self.query(
            """SELECT * FROM facts
            WHERE key LIKE ? OR value LIKE ?
            LIMIT ?""",
            (f"%{keyword}%", f"%{keyword}%", limit),
        )

    @staticmethod
    def _parse_numeric(value: str) -> Optional[float]:
        """Parse a string value into a numeric value, handling common formats."""
        if not value:
            return None
        # Remove common formatting
        cleaned = re.sub(r"[,$%\s]", "", value.strip())
        # Handle parentheses for negative numbers
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        # Handle B/M/K suffixes
        multiplier = 1.0
        if cleaned.upper().endswith("B"):
            multiplier = 1e9
            cleaned = cleaned[:-1]
        elif cleaned.upper().endswith("M"):
            multiplier = 1e6
            cleaned = cleaned[:-1]
        elif cleaned.upper().endswith("K"):
            multiplier = 1e3
            cleaned = cleaned[:-1]
        try:
            return float(cleaned) * multiplier
        except (ValueError, OverflowError):
            return None

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


def structured_query(
    fact_table: FactTable,
    query: str,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute a structured query against the FactTable.

    Supports:
    - Direct SQL queries (if query starts with SELECT)
    - Keyword search (otherwise)

    Args:
        fact_table: FactTable instance.
        query: SQL query or keyword search term.
        document_id: Optional document filter.

    Returns:
        List of result dicts.
    """
    if query.strip().upper().startswith("SELECT"):
        return fact_table.query(query)
    else:
        return fact_table.search_facts(query, document_id)


# ═══════════════════════════════════════════════════════════════════════════
# Provenance Builder
# ═══════════════════════════════════════════════════════════════════════════


def build_provenance_chain(
    document_id: str,
    document_name: str,
    claim: str,
    source_ldus: List[LDU],
    confidence: float = 0.0,
) -> ProvenanceChain:
    """Build a ProvenanceChain from source LDUs.

    Args:
        document_id: Document identifier.
        document_name: Human-readable filename.
        claim: The fact/answer being cited.
        source_ldus: LDUs that support this claim.
        confidence: Overall confidence for the chain.

    Returns:
        ProvenanceChain with source citations.
    """
    citations = []
    for ldu in source_ldus:
        citation = SourceCitation(
            page_number=ldu.page_number,
            bbox=ldu.bbox,
            text_snippet=ldu.content[:500],
            content_hash=ldu.content_hash,
            section_heading=ldu.section_heading,
        )
        citations.append(citation)

    if not citations:
        # Create a minimal citation for "not found" case
        citations = [
            SourceCitation(
                page_number=1,
                text_snippet="No matching content found.",
                content_hash=hashlib.sha256(b"not_found").hexdigest()[:16],
            )
        ]
        confidence = 0.0

    return ProvenanceChain(
        document_id=document_id,
        document_name=document_name,
        claim=claim,
        citations=citations,
        confidence=confidence,
        verification_status="verified" if confidence > 0.7 else "unverified",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Audit Mode
# ═══════════════════════════════════════════════════════════════════════════


class AuditMode:
    """Claim verification system.

    Given a claim, searches the document corpus to either:
    - Verify with source citation (ProvenanceChain)
    - Flag as "not found / unverifiable"
    """

    def __init__(
        self,
        vector_store: VectorStore,
        fact_table: FactTable,
        ldu_cache: Optional[Dict[str, List[LDU]]] = None,
    ):
        self.vector_store = vector_store
        self.fact_table = fact_table
        self.ldu_cache = ldu_cache or {}

    def verify_claim(
        self,
        claim: str,
        document_id: str,
        document_name: str,
    ) -> ProvenanceChain:
        """Verify a claim against the document corpus.

        Strategy:
        1. Search FactTable for exact numerical matches.
        2. Search VectorStore for semantic similarity.
        3. Build ProvenanceChain from best matches.

        Args:
            claim: The claim to verify (e.g., "Revenue was $4.2B in Q3")
            document_id: Document to verify against.
            document_name: Document filename.

        Returns:
            ProvenanceChain with verification status.
        """
        # Step 1: Check FactTable for numerical claims
        fact_results = self.fact_table.search_facts(claim[:100], document_id, limit=5)
        valid_facts = [f for f in fact_results if "error" not in f]

        # Step 2: Semantic search
        search_results = self.vector_store.search(claim, document_id, top_k=5)

        # Step 3: Build source LDUs from results
        source_ldus = []
        confidence = 0.0

        # Convert search results to source LDUs
        for result in search_results:
            meta = result.get("metadata", {})
            ldu = LDU(
                ldu_id=result["ldu_id"],
                document_id=document_id,
                ldu_type=meta.get("ldu_type", "paragraph"),
                content=result.get("content", ""),
                page_number=meta.get("page_number", 1),
                content_hash=meta.get("content_hash", ""),
                section_heading=meta.get("section_heading"),
                extraction_strategy="retrieved",
                confidence=meta.get("confidence", 0.5),
            )
            source_ldus.append(ldu)

        # Calculate confidence based on match quality
        if valid_facts:
            confidence = min(0.95, 0.7 + 0.05 * len(valid_facts))
        elif search_results:
            # Use inverse distance as confidence
            best_distance = search_results[0].get("distance", 1.0)
            confidence = max(0.3, min(0.9, 1.0 - best_distance))
        else:
            confidence = 0.0

        # Determine verification status
        if confidence >= 0.7:
            status = "verified"
        elif confidence >= 0.3:
            status = "unverified"
        else:
            status = "unverifiable"

        chain = build_provenance_chain(
            document_id=document_id,
            document_name=document_name,
            claim=claim,
            source_ldus=source_ldus,
            confidence=confidence,
        )
        chain.verification_status = status

        logger.info(
            "AuditMode: Claim '%s...' → %s (confidence %.2f, %d citations)",
            claim[:50],
            status,
            confidence,
            len(chain.citations),
        )

        return chain
