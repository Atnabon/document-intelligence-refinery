"""PageIndex Builder Agent — Phase 4 of the Document Intelligence Refinery.

Builds a hierarchical section tree (PageIndex) from extracted LDUs, providing
deterministic navigation over documents. The tree enables section-specific
retrieval without relying on vector similarity search.

Each PageNode carries:
- Title, page range, child sections
- LDU IDs belonging to this section
- LLM-generated summary (2-3 sentences)
- Metadata: data types present (tables, figures, etc.)
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.ldu import LDU, LDUType
from src.models.page_index import PageIndex, PageNode

logger = logging.getLogger(__name__)


class PageIndexBuilder:
    """Builds a PageIndex tree from a list of LDUs.

    The builder:
    1. Identifies heading LDUs and infers hierarchy from heading level.
    2. Assigns non-heading LDUs to their nearest ancestor section.
    3. Computes page ranges for each section.
    4. Generates LLM summaries for each section (optional, requires API key).
    5. Annotates metadata: data_types_present, key_entities.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.vlm_cfg = self.config.get("vlm", {})
        self.output_dir = Path(
            self.config.get("output", {}).get("pageindex_dir", ".refinery/pageindex")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        document_id: str,
        ldus: List[LDU],
        document_title: Optional[str] = None,
        generate_summaries: bool = True,
    ) -> PageIndex:
        """Build a PageIndex from a list of LDUs.

        Args:
            document_id: Document identifier.
            ldus: All LDUs for the document (should be chunked already).
            document_title: Title for the root node (defaults to document_id).
            generate_summaries: Whether to generate LLM summaries.

        Returns:
            Complete PageIndex with hierarchical section tree.
        """
        if not ldus:
            root = PageNode(
                node_id="root",
                title=document_title or document_id,
                level=0,
                page_start=1,
                page_end=1,
            )
            return PageIndex(
                document_id=document_id,
                root=root,
                total_sections=1,
                max_depth=0,
            )

        logger.info(
            "PageIndexBuilder: Building index for %s (%d LDUs)",
            document_id,
            len(ldus),
        )

        # Step 1: Extract heading hierarchy
        headings = self._extract_headings(ldus)

        # Step 2: Build tree structure
        all_pages = [ldu.page_number for ldu in ldus]
        min_page = min(all_pages) if all_pages else 1
        max_page = max(all_pages) if all_pages else 1

        root = PageNode(
            node_id="root",
            title=document_title or document_id,
            level=0,
            page_start=min_page,
            page_end=max_page,
        )

        if headings:
            root.children = self._build_children(headings, ldus, base_level=1)
        else:
            # No headings found — create page-range sections
            root.children = self._create_page_sections(ldus, pages_per_section=10)

        # Step 3: Assign LDU IDs to sections
        self._assign_ldu_ids(root, ldus)

        # Step 4: Compute metadata
        self._compute_metadata(root, ldus)

        # Step 5: Generate summaries
        if generate_summaries:
            self._generate_summaries(root, ldus)

        # Step 6: Count sections and depth
        total_sections = self._count_nodes(root)
        max_depth = self._compute_depth(root)

        index = PageIndex(
            document_id=document_id,
            root=root,
            total_sections=total_sections,
            max_depth=max_depth,
        )

        logger.info(
            "PageIndexBuilder: Built index with %d sections (depth %d)",
            total_sections,
            max_depth,
        )

        return index

    def save(self, index: PageIndex) -> Path:
        """Save PageIndex to JSON file."""
        path = self.output_dir / f"{index.document_id}_pageindex.json"
        with open(path, "w") as f:
            f.write(index.model_dump_json(indent=2))
        logger.info("PageIndex saved: %s", path)
        return path

    def navigate(
        self, index: PageIndex, topic: str, top_k: int = 3
    ) -> List[PageNode]:
        """Navigate the PageIndex tree to find sections relevant to a topic.

        Uses keyword matching against section titles, summaries, and metadata.
        This is a deterministic, non-probabilistic traversal.

        Args:
            index: The PageIndex to search.
            topic: Topic string to search for.
            top_k: Number of top sections to return.

        Returns:
            List of most relevant PageNode objects.
        """
        topic_lower = topic.lower()
        topic_words = set(re.findall(r"\w+", topic_lower))

        scored_nodes: List[Tuple[float, PageNode]] = []
        self._score_node(index.root, topic_lower, topic_words, scored_nodes)

        # Sort by score descending
        scored_nodes.sort(key=lambda x: x[0], reverse=True)

        return [node for _, node in scored_nodes[:top_k]]

    def _score_node(
        self,
        node: PageNode,
        topic_lower: str,
        topic_words: set,
        results: List[Tuple[float, PageNode]],
    ) -> None:
        """Recursively score nodes for relevance to topic."""
        score = 0.0

        # Score title match
        title_lower = node.title.lower()
        title_words = set(re.findall(r"\w+", title_lower))
        word_overlap = len(topic_words & title_words)
        score += word_overlap * 3.0

        # Exact substring match in title
        if topic_lower in title_lower:
            score += 5.0

        # Score summary match
        if node.summary:
            summary_lower = node.summary.lower()
            summary_words = set(re.findall(r"\w+", summary_lower))
            score += len(topic_words & summary_words) * 1.0
            if topic_lower in summary_lower:
                score += 3.0

        # Score key_entities match
        for entity in node.key_entities:
            if any(w in entity.lower() for w in topic_words):
                score += 2.0

        # Score data_types_present match
        for dtype in node.data_types_present:
            if any(w in dtype.lower() for w in topic_words):
                score += 1.5

        # Score metadata match
        for key, value in node.metadata.items():
            if any(w in value.lower() for w in topic_words):
                score += 1.0

        # Boost leaf nodes (more specific)
        if not node.children:
            score *= 1.2

        if score > 0:
            results.append((score, node))

        for child in node.children:
            self._score_node(child, topic_lower, topic_words, results)

    def _extract_headings(
        self, ldus: List[LDU]
    ) -> List[Tuple[LDU, int]]:
        """Extract heading LDUs and assign hierarchy levels.

        Returns list of (heading_ldu, level) tuples.
        """
        headings = []
        for ldu in ldus:
            if ldu.ldu_type == LDUType.HEADING:
                level = self._infer_heading_level(ldu)
                headings.append((ldu, level))

        return headings

    def _infer_heading_level(self, ldu: LDU) -> int:
        """Infer heading level from content patterns.

        Uses heuristics:
        - Numbered sections: "1." = level 1, "1.1" = level 2, "1.1.1" = level 3
        - ALL CAPS = level 1
        - Title Case short text = level 2
        - Other = level 3
        """
        text = ldu.content.strip()

        # Check numbered patterns
        num_match = re.match(r"^(\d+(?:\.\d+)*)[.\s)\-]", text)
        if num_match:
            parts = num_match.group(1).split(".")
            return min(len(parts), 4)

        # Check roman numeral patterns
        roman_match = re.match(
            r"^(?:Part|Chapter|Section)\s+(?:[IVXLCDM]+|\d+)", text, re.IGNORECASE
        )
        if roman_match:
            return 1

        # ALL CAPS (excluding short labels)
        if text.isupper() and len(text) > 5:
            return 1

        # Mixed case short heading
        if len(text) < 80:
            return 2

        return 3

    def _build_children(
        self,
        headings: List[Tuple[LDU, int]],
        ldus: List[LDU],
        base_level: int,
    ) -> List[PageNode]:
        """Build child nodes from heading list using a stack-based approach."""
        if not headings:
            return []

        # Group by level — build tree iteratively
        nodes: List[PageNode] = []
        stack: List[Tuple[PageNode, int]] = []  # (node, heading_level)

        for i, (heading_ldu, level) in enumerate(headings):
            # Determine page range
            page_start = heading_ldu.page_number
            if i + 1 < len(headings):
                page_end = headings[i + 1][0].page_number
                if headings[i + 1][0].page_number > page_start:
                    page_end = headings[i + 1][0].page_number - 1
                else:
                    page_end = page_start
            else:
                # Last heading — extends to end of document
                all_pages = [l.page_number for l in ldus]
                page_end = max(all_pages) if all_pages else page_start

            page_end = max(page_end, page_start)

            node = PageNode(
                node_id=f"s{i + 1}",
                title=heading_ldu.content.strip()[:200],
                level=level,
                page_start=page_start,
                page_end=page_end,
            )

            # Pop stack to find parent
            while stack and stack[-1][1] >= level:
                stack.pop()

            if stack:
                # Add as child of the top of stack
                parent_node = stack[-1][0]
                parent_node.children.append(node)
            else:
                # Top-level section
                nodes.append(node)

            stack.append((node, level))

        return nodes

    def _create_page_sections(
        self, ldus: List[LDU], pages_per_section: int = 10
    ) -> List[PageNode]:
        """Create page-range-based sections when no headings are found."""
        all_pages = sorted(set(ldu.page_number for ldu in ldus))
        if not all_pages:
            return []

        sections = []
        for i in range(0, len(all_pages), pages_per_section):
            chunk_pages = all_pages[i : i + pages_per_section]
            node = PageNode(
                node_id=f"pg_{chunk_pages[0]}_{chunk_pages[-1]}",
                title=f"Pages {chunk_pages[0]}–{chunk_pages[-1]}",
                level=1,
                page_start=chunk_pages[0],
                page_end=chunk_pages[-1],
            )
            sections.append(node)

        return sections

    def _assign_ldu_ids(self, node: PageNode, ldus: List[LDU]) -> None:
        """Assign LDU IDs to the most specific section they belong to."""
        if node.children:
            for child in node.children:
                self._assign_ldu_ids(child, ldus)
        else:
            # Leaf node — assign LDUs by page range
            for ldu in ldus:
                if node.page_start <= ldu.page_number <= node.page_end:
                    node.ldu_ids.append(ldu.ldu_id)

    def _compute_metadata(self, node: PageNode, ldus: List[LDU]) -> None:
        """Compute metadata for each node: data_types_present, key_entities."""
        if node.children:
            for child in node.children:
                self._compute_metadata(child, ldus)
            # Aggregate children data
            all_data_types = set()
            all_entities = set()
            for child in node.children:
                all_data_types.update(child.data_types_present)
                all_entities.update(child.key_entities)
            node.data_types_present = sorted(all_data_types)
            node.key_entities = sorted(all_entities)[:20]  # Cap at 20 for parent
        else:
            # Leaf node — compute from LDUs
            section_ldus = [
                l for l in ldus if l.ldu_id in node.ldu_ids
            ]

            # Populate data_types_present
            data_types = set()
            for ldu in section_ldus:
                dtype = ldu.ldu_type if isinstance(ldu.ldu_type, str) else ldu.ldu_type.value
                if dtype in ("table", "figure", "list", "key_value"):
                    data_types.add(dtype)
            node.data_types_present = sorted(data_types)

            # Populate key_entities via pattern-based extraction
            entities = self._extract_entities(section_ldus)
            node.key_entities = entities[:10]  # Cap at 10 per leaf

            # Keep ldu_count in metadata
            node.metadata["ldu_count"] = str(len(section_ldus))

    @staticmethod
    def _extract_entities(ldus: List[LDU]) -> List[str]:
        """Extract named entities from LDU content using pattern matching.

        Detects: organizations (ALL CAPS acronyms), monetary amounts,
        dates/years, percentages, and proper nouns.
        """
        entities: Dict[str, int] = defaultdict(int)  # entity → count
        combined_text = " ".join(ldu.content[:500] for ldu in ldus[:10])

        # Monetary amounts: $1.2B, ETB 245,678, USD 1,000
        for m in re.findall(
            r"(?:USD|ETB|EUR|GBP|\$|£|€)\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|[BMK])?",
            combined_text,
            re.IGNORECASE,
        ):
            entities[m.strip()] += 1

        # Percentages: 15.2%, 3 percent
        for m in re.findall(r"\d+(?:\.\d+)?%", combined_text):
            entities[m] += 1

        # Years: 2018-2024 standalone
        for m in re.findall(r"\b((?:19|20)\d{2}(?:[/-]\d{2,4})?)\b", combined_text):
            entities[m] += 1

        # Acronyms / Organization names (2-6 uppercase letters)
        for m in re.findall(r"\b([A-Z]{2,6})\b", combined_text):
            if m not in ("THE", "AND", "FOR", "NOT", "ARE", "WAS",
                         "HAS", "ALL", "BUT", "FROM", "WITH"):
                entities[m] += 1

        # Sort by frequency and return top entities
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
        return [e[0] for e in sorted_entities]

    def _generate_summaries(self, node: PageNode, ldus: List[LDU]) -> None:
        """Generate text summaries for each section node.

        Uses heuristic summarization (first N chars of section content)
        as the default. Falls back to this if no LLM API key is available.
        LLM-based summarization can be enabled via config.
        """
        # Try LLM summarization first
        api_key = os.environ.get("OPENAI_API_KEY")
        use_llm = api_key and self.vlm_cfg.get("provider") in ("openai", "google")

        if node.children:
            for child in node.children:
                self._generate_summaries(child, ldus)

            # Summarize parent from children summaries
            child_summaries = [
                c.summary for c in node.children if c.summary
            ]
            if child_summaries:
                combined = " ".join(child_summaries[:5])
                node.summary = self._truncate_summary(combined, max_len=300)
            else:
                node.summary = f"Section covering pages {node.page_start}–{node.page_end}."
        else:
            # Leaf node — summarize from LDU content
            section_ldus = [l for l in ldus if l.ldu_id in node.ldu_ids]
            if section_ldus:
                # Extract first meaningful content
                content_parts = []
                for ldu in section_ldus[:5]:
                    text = ldu.content.strip()
                    if text and len(text) > 20:
                        content_parts.append(text)

                if content_parts:
                    combined = " ".join(content_parts)
                    if use_llm:
                        node.summary = self._llm_summarize(
                            node.title, combined[:2000]
                        )
                    else:
                        node.summary = self._truncate_summary(combined, max_len=300)
                else:
                    node.summary = f"Section '{node.title}' on page {node.page_start}."
            else:
                node.summary = f"Section covering pages {node.page_start}–{node.page_end}."

    def _llm_summarize(self, title: str, content: str) -> str:
        """Generate a 2-3 sentence summary using an LLM."""
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document summarizer. Generate a concise 2-3 sentence summary "
                            "of the following document section. Focus on key facts, figures, and findings."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Section: {title}\n\nContent:\n{content[:2000]}",
                    },
                ],
                max_tokens=150,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("LLM summarization failed: %s. Using heuristic.", e)
            return self._truncate_summary(content, max_len=300)

    @staticmethod
    def _truncate_summary(text: str, max_len: int = 300) -> str:
        """Create a truncated summary from text."""
        text = text.strip()
        if len(text) <= max_len:
            return text
        # Cut at sentence boundary
        truncated = text[:max_len]
        last_period = truncated.rfind(".")
        if last_period > max_len // 2:
            return truncated[: last_period + 1]
        return truncated + "..."

    def _count_nodes(self, node: PageNode) -> int:
        """Count total nodes in the tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _compute_depth(self, node: PageNode) -> int:
        """Compute maximum depth of the tree."""
        if not node.children:
            return 0
        return 1 + max(self._compute_depth(c) for c in node.children)
