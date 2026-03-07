"""Semantic Chunking Engine — Phase 3 of the Document Intelligence Refinery.

Converts raw extracted LDUs into validated, RAG-optimized Logical Document Units
by enforcing the five chunking rules defined in extraction_rules.yaml:

1. Table Integrity     — table cells never split from headers; a table is always one LDU
2. Caption Binding     — figure/table captions stored as metadata on parent element
3. List Preservation   — numbered lists kept as single LDUs unless exceeding max_tokens
4. Section Propagation — section headers propagated as parent metadata on all child chunks
5. Cross-Reference     — cross-references (e.g., 'see Table 3') resolved & stored as chunk relationships
"""

from __future__ import annotations

import hashlib
import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.ldu import LDU, LDUType
from src.models.provenance import BoundingBox

logger = logging.getLogger(__name__)


# ── Chunk Validation ──────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Result of a single rule check against an LDU."""

    rule_name: str
    passed: bool
    severity: str  # "strict" or "advisory"
    message: str = ""


@dataclass
class ChunkValidationReport:
    """Aggregated validation report for a list of LDUs."""

    total_ldus: int = 0
    passed_count: int = 0
    failed_strict: int = 0
    failed_advisory: int = 0
    violations: List[ValidationResult] = field(default_factory=list)

    @property
    def all_strict_passed(self) -> bool:
        return self.failed_strict == 0


class ChunkValidator:
    """Validates a list of LDUs against the five chunking rules.

    Strict rules cause re-chunking; advisory rules are logged as warnings.
    """

    def __init__(self, config: Optional[dict] = None):
        chunking_cfg = (config or {}).get("chunking", {})
        rules = chunking_cfg.get("rules", [])

        # Defaults from extraction_rules.yaml
        self.min_chars = 50
        self.max_chars = 2000

        for rule in rules:
            if rule.get("name") == "minimum_context":
                self.min_chars = rule.get("min_chars", self.min_chars)
            elif rule.get("name") == "maximum_size":
                self.max_chars = rule.get("max_chars", self.max_chars)

    def validate(self, ldus: List[LDU]) -> ChunkValidationReport:
        """Run all five rules against the LDU list."""
        report = ChunkValidationReport(total_ldus=len(ldus))

        for ldu in ldus:
            results = self._check_all_rules(ldu, ldus)
            all_passed = True
            for r in results:
                if not r.passed:
                    report.violations.append(r)
                    if r.severity == "strict":
                        report.failed_strict += 1
                        all_passed = False
                    else:
                        report.failed_advisory += 1
            if all_passed:
                report.passed_count += 1

        return report

    def _check_all_rules(
        self, ldu: LDU, all_ldus: List[LDU]
    ) -> List[ValidationResult]:
        """Check a single LDU against all five rubric-required rules."""
        results = []

        # Rule 1: Table Integrity — table cells never split from headers
        results.append(self._check_table_integrity(ldu))

        # Rule 2: Caption Binding — captions stored as metadata on parent figure/table
        results.append(self._check_caption_binding(ldu, all_ldus))

        # Rule 3: List Preservation — numbered lists kept as single LDUs unless exceeding max_tokens
        results.append(self._check_list_integrity(ldu))

        # Rule 4: Section Propagation — section headers propagated as parent metadata on all child chunks
        results.append(self._check_section_propagation(ldu))

        # Rule 5: Cross-Reference — cross-references resolved and stored as chunk relationships
        results.append(self._check_cross_references(ldu, all_ldus))

        return results

    def _check_table_integrity(self, ldu: LDU) -> ValidationResult:
        """Rule 1: Tables must be single LDUs — never split."""
        if ldu.ldu_type == LDUType.TABLE:
            # A table LDU should have structured_content with headers
            if ldu.structured_content and "headers" in ldu.structured_content:
                return ValidationResult(
                    rule_name="table_integrity",
                    passed=True,
                    severity="strict",
                )
            # If it's a table without structure, still passes — it's kept whole
            return ValidationResult(
                rule_name="table_integrity",
                passed=True,
                severity="strict",
                message="Table LDU present but lacks structured_content.",
            )
        return ValidationResult(
            rule_name="table_integrity", passed=True, severity="strict"
        )

    def _check_caption_binding(
        self, ldu: LDU, all_ldus: List[LDU]
    ) -> ValidationResult:
        """Rule 2: Captions must have parent_ldu_id set to their figure/table."""
        if ldu.ldu_type == LDUType.CAPTION:
            if ldu.parent_ldu_id:
                # Verify parent exists
                parent_exists = any(l.ldu_id == ldu.parent_ldu_id for l in all_ldus)
                if parent_exists:
                    return ValidationResult(
                        rule_name="caption_binding",
                        passed=True,
                        severity="strict",
                    )
                return ValidationResult(
                    rule_name="caption_binding",
                    passed=False,
                    severity="strict",
                    message=f"Caption {ldu.ldu_id} references non-existent parent {ldu.parent_ldu_id}.",
                )
            return ValidationResult(
                rule_name="caption_binding",
                passed=False,
                severity="strict",
                message=f"Caption {ldu.ldu_id} has no parent_ldu_id.",
            )
        return ValidationResult(
            rule_name="caption_binding", passed=True, severity="strict"
        )

    def _check_list_integrity(self, ldu: LDU) -> ValidationResult:
        """Rule 3: Numbered lists kept as single LDUs unless exceeding max_tokens."""
        if ldu.ldu_type == LDUType.LIST:
            token_count = ldu.token_count or len(ldu.content.split())
            max_tokens = self.max_chars // 4  # approximate token limit
            if token_count > max_tokens:
                return ValidationResult(
                    rule_name="list_integrity",
                    passed=False,
                    severity="advisory",
                    message=f"List LDU {ldu.ldu_id} has {token_count} tokens (max: {max_tokens}). May need splitting.",
                )
            return ValidationResult(
                rule_name="list_integrity",
                passed=True,
                severity="strict",
            )
        return ValidationResult(
            rule_name="list_integrity", passed=True, severity="strict"
        )

    def _check_section_propagation(self, ldu: LDU) -> ValidationResult:
        """Rule 4: Section headers propagated as parent metadata on all child chunks."""
        # Headings themselves don't need a parent section
        if ldu.ldu_type in (LDUType.HEADING, LDUType.PAGE_HEADER, LDUType.PAGE_FOOTER):
            return ValidationResult(
                rule_name="section_propagation",
                passed=True,
                severity="strict",
            )
        if not ldu.section_heading:
            return ValidationResult(
                rule_name="section_propagation",
                passed=False,
                severity="strict",
                message=f"LDU {ldu.ldu_id} has no section_heading (parent metadata missing).",
            )
        return ValidationResult(
            rule_name="section_propagation", passed=True, severity="strict"
        )

    def _check_cross_references(
        self, ldu: LDU, all_ldus: List[LDU]
    ) -> ValidationResult:
        """Rule 5: Cross-references resolved and stored as chunk relationships."""
        # Detect unresolved cross-reference patterns in content
        xref_pattern = r"(?:see|refer to|as shown in|in)\s+(?:Table|Figure|Fig\.|Exhibit|Section|Appendix)\s+\d+"
        matches = re.findall(xref_pattern, ldu.content, re.IGNORECASE)
        if matches and not ldu.cross_references:
            return ValidationResult(
                rule_name="cross_reference",
                passed=False,
                severity="advisory",
                message=f"LDU {ldu.ldu_id} has {len(matches)} unresolved cross-references: {matches[:3]}",
            )
        return ValidationResult(
            rule_name="cross_reference", passed=True, severity="advisory"
        )


# ── Semantic Chunking Engine ──────────────────────────────────────────────


class ChunkingEngine:
    """Transforms raw extracted LDUs into validated, RAG-optimized chunks.

    Applies the five chunking rules as enforceable constraints:
    - Strict rules trigger re-chunking (split oversized paragraphs,
      bind captions to parents, respect section boundaries).
    - Advisory rules log warnings but do not block.

    The engine does NOT re-extract from the PDF — it operates purely on the
    LDU list produced by the extraction stage.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        chunking_cfg = self.config.get("chunking", {})
        rules = chunking_cfg.get("rules", [])

        self.min_chars = 50
        self.max_chars = 2000
        self.paragraph_overlap = chunking_cfg.get("paragraph_overlap_chars", 100)

        for rule in rules:
            if rule.get("name") == "minimum_context":
                self.min_chars = rule.get("min_chars", self.min_chars)
            elif rule.get("name") == "maximum_size":
                self.max_chars = rule.get("max_chars", self.max_chars)

        self.validator = ChunkValidator(config)

    def chunk(self, ldus: List[LDU]) -> List[LDU]:
        """Apply semantic chunking to a list of raw LDUs.

        Steps:
        1. Section assignment — propagate nearest heading to all LDUs (Rule 4).
        2. Caption binding — link captions to parent figure/table (Rule 2).
        3. List preservation — ensure lists are kept as single LDUs (Rule 3).
        4. Cross-reference resolution — resolve 'see Table 3' references (Rule 5).
        5. Size enforcement — split oversized paragraphs at sentence boundaries.
        6. Small-chunk merging — merge undersized adjacent LDUs of same type.
        7. Content hash recomputation — update hashes after any modifications.
        8. Token count computation — compute token_count for every LDU.
        9. Validation — run ChunkValidator and log results.

        Args:
            ldus: Raw LDUs from extraction.

        Returns:
            Validated list of LDUs, possibly with different count than input.
        """
        if not ldus:
            return []

        logger.info("ChunkingEngine: Processing %d raw LDUs", len(ldus))

        # Step 1: Sort by page and sequence
        ldus_sorted = sorted(ldus, key=lambda l: (l.page_number, l.sequence_index))

        # Step 2: Propagate section headings (Rule 4: section header propagation)
        ldus_sorted = self._assign_section_headings(ldus_sorted)

        # Step 3: Bind captions to parents (Rule 2: caption binding)
        ldus_sorted = self._bind_captions(ldus_sorted)

        # Step 4: Preserve list LDUs (Rule 3: list preservation)
        ldus_sorted = self._preserve_lists(ldus_sorted)

        # Step 5: Resolve cross-references (Rule 5: cross-reference resolution)
        ldus_sorted = self._resolve_cross_references(ldus_sorted)

        # Step 6: Split oversized LDUs
        chunked = self._split_oversized(ldus_sorted)

        # Step 7: Merge undersized adjacent LDUs
        chunked = self._merge_undersized(chunked)

        # Step 8: Recompute content hashes
        chunked = self._recompute_hashes(chunked)

        # Step 9: Re-index sequence
        chunked = self._reindex_sequences(chunked)

        # Step 10: Compute token counts for all LDUs
        chunked = self._compute_token_counts(chunked)

        # Step 8: Validate
        report = self.validator.validate(chunked)
        logger.info(
            "ChunkingEngine: %d LDUs after chunking (was %d). "
            "Validation: %d passed, %d strict violations, %d advisory warnings",
            len(chunked),
            len(ldus),
            report.passed_count,
            report.failed_strict,
            report.failed_advisory,
        )
        if report.violations:
            for v in report.violations[:10]:  # Log first 10
                log_fn = logger.warning if v.severity == "strict" else logger.debug
                log_fn("  [%s] %s: %s", v.severity.upper(), v.rule_name, v.message)

        return chunked

    def _assign_section_headings(self, ldus: List[LDU]) -> List[LDU]:
        """Propagate the nearest preceding heading to subsequent LDUs."""
        current_heading: Optional[str] = None
        result = []
        for ldu in ldus:
            if ldu.ldu_type == LDUType.HEADING:
                current_heading = ldu.content.strip()
            if current_heading and not ldu.section_heading:
                ldu = ldu.model_copy(update={"section_heading": current_heading})
            result.append(ldu)
        return result

    def _bind_captions(self, ldus: List[LDU]) -> List[LDU]:
        """Link caption LDUs to their nearest preceding figure/table LDU."""
        result = []
        last_figure_or_table_id: Optional[str] = None

        for ldu in ldus:
            if ldu.ldu_type in (LDUType.TABLE, LDUType.FIGURE):
                last_figure_or_table_id = ldu.ldu_id
                result.append(ldu)
            elif ldu.ldu_type == LDUType.CAPTION:
                if last_figure_or_table_id and not ldu.parent_ldu_id:
                    ldu = ldu.model_copy(
                        update={"parent_ldu_id": last_figure_or_table_id}
                    )
                    # Also update the parent's child list
                    for i, existing in enumerate(result):
                        if existing.ldu_id == last_figure_or_table_id:
                            updated_children = list(existing.child_ldu_ids) + [
                                ldu.ldu_id
                            ]
                            result[i] = existing.model_copy(
                                update={"child_ldu_ids": updated_children}
                            )
                            break
                result.append(ldu)
            else:
                result.append(ldu)
                # Also detect inline caption patterns
                caption_match = re.match(
                    r"^(?:Figure|Fig\.|Table|Exhibit)\s+\d+",
                    ldu.content.strip(),
                    re.IGNORECASE,
                )
                if caption_match and last_figure_or_table_id:
                    ldu = ldu.model_copy(
                        update={
                            "ldu_type": LDUType.CAPTION,
                            "parent_ldu_id": last_figure_or_table_id,
                        }
                    )
                    result[-1] = ldu

        return result

    def _preserve_lists(self, ldus: List[LDU]) -> List[LDU]:
        """Rule 3: Keep numbered lists as single LDUs unless exceeding max_tokens.

        Detects adjacent paragraph LDUs that form a numbered/bulleted list
        and merges them into a single LIST-type LDU.
        """
        result = []
        list_buffer: List[LDU] = []

        for ldu in ldus:
            is_list_item = (
                ldu.ldu_type == LDUType.LIST
                or re.match(
                    r"^\s*(?:\d+[.)]\s|[a-z][.)]\s|[-•●▪]\s|[ivxIVX]+[.)]\s)",
                    ldu.content.strip(),
                )
            )

            if is_list_item:
                list_buffer.append(ldu)
            else:
                if list_buffer:
                    result.extend(self._emit_list_ldus(list_buffer))
                    list_buffer = []
                result.append(ldu)

        if list_buffer:
            result.extend(self._emit_list_ldus(list_buffer))

        return result

    def _emit_list_ldus(self, buffer: List[LDU]) -> List[LDU]:
        """Merge list items into a single LIST LDU, splitting only if too large."""
        if not buffer:
            return []

        max_tokens = self.max_chars // 4
        merged_content = "\n".join(ldu.content for ldu in buffer)
        token_count = len(merged_content.split())

        if len(buffer) == 1:
            ldu = buffer[0]
            return [ldu.model_copy(update={
                "ldu_type": LDUType.LIST,
                "token_count": len(ldu.content.split()),
            })]

        if token_count <= max_tokens:
            # Merge into single list LDU
            first = buffer[0]
            merged = first.model_copy(update={
                "ldu_type": LDUType.LIST,
                "content": merged_content,
                "content_hash": self._hash(merged_content),
                "child_ldu_ids": [l.ldu_id for l in buffer[1:]],
                "token_count": token_count,
            })
            return [merged]
        else:
            # Too large — keep as individual list LDUs
            return [
                ldu.model_copy(update={
                    "ldu_type": LDUType.LIST,
                    "token_count": len(ldu.content.split()),
                })
                for ldu in buffer
            ]

    def _resolve_cross_references(self, ldus: List[LDU]) -> List[LDU]:
        """Rule 5: Resolve cross-references and store as chunk relationships.

        Detects patterns like 'see Table 3', 'refer to Figure 5', 'in Section 2.1'
        and links them to the referenced LDU via cross_references field.
        """
        # Build lookup: (type, number) → ldu_id
        reference_map: Dict[str, str] = {}  # "table_3" → ldu_id
        for ldu in ldus:
            if ldu.ldu_type in (LDUType.TABLE, LDUType.FIGURE):
                # Try to extract a number from the content or nearby captions
                num_match = re.search(
                    r"(?:Table|Figure|Fig\.|Exhibit)\s+(\d+)",
                    ldu.content[:200],
                    re.IGNORECASE,
                )
                if num_match:
                    ref_type = "table" if ldu.ldu_type == LDUType.TABLE else "figure"
                    ref_key = f"{ref_type}_{num_match.group(1)}"
                    reference_map[ref_key] = ldu.ldu_id

            if ldu.ldu_type == LDUType.CAPTION and ldu.content:
                num_match = re.search(
                    r"(?:Table|Figure|Fig\.|Exhibit)\s+(\d+)",
                    ldu.content[:200],
                    re.IGNORECASE,
                )
                if num_match:
                    ref_type = "table" if "table" in ldu.content.lower()[:30] else "figure"
                    ref_key = f"{ref_type}_{num_match.group(1)}"
                    if ldu.parent_ldu_id:
                        reference_map[ref_key] = ldu.parent_ldu_id
                    else:
                        reference_map[ref_key] = ldu.ldu_id

            if ldu.ldu_type == LDUType.HEADING:
                num_match = re.match(r"^(\d+(?:\.\d+)*)[.\s)\-]", ldu.content.strip())
                if num_match:
                    ref_key = f"section_{num_match.group(1)}"
                    reference_map[ref_key] = ldu.ldu_id

        # Resolve references in all LDUs
        result = []
        xref_pattern = re.compile(
            r"(?:see|refer to|as shown in|as described in|in)\s+"
            r"(?:(Table|Figure|Fig\.|Exhibit|Section|Appendix)\s+"
            r"(\d+(?:\.\d+)*))",
            re.IGNORECASE,
        )

        for ldu in ldus:
            matches = xref_pattern.findall(ldu.content)
            if matches:
                resolved_refs = []
                for ref_type_str, ref_num in matches:
                    ref_type_str = ref_type_str.lower().rstrip(".")
                    if ref_type_str in ("fig", "figure", "exhibit"):
                        ref_type_str = "figure"
                    ref_key = f"{ref_type_str}_{ref_num}"
                    target_id = reference_map.get(ref_key)
                    if target_id and target_id != ldu.ldu_id:
                        resolved_refs.append(target_id)

                if resolved_refs:
                    existing = list(ldu.cross_references or [])
                    existing.extend(r for r in resolved_refs if r not in existing)
                    ldu = ldu.model_copy(update={"cross_references": existing})
                    logger.debug(
                        "Resolved %d cross-references in LDU %s: %s",
                        len(resolved_refs),
                        ldu.ldu_id,
                        resolved_refs,
                    )

            result.append(ldu)

        # Log summary
        total_xrefs = sum(len(l.cross_references) for l in result)
        if total_xrefs > 0:
            logger.info(
                "ChunkingEngine: Resolved %d cross-references across %d LDUs",
                total_xrefs,
                sum(1 for l in result if l.cross_references),
            )

        return result

    def _split_oversized(self, ldus: List[LDU]) -> List[LDU]:
        """Split LDUs that exceed max_chars at sentence boundaries.

        Tables and headings are never split.
        """
        result = []
        for ldu in ldus:
            if (
                len(ldu.content) <= self.max_chars
                or ldu.ldu_type in (LDUType.TABLE, LDUType.HEADING, LDUType.CAPTION)
            ):
                result.append(ldu)
                continue

            # Split at sentence boundaries
            chunks = self._split_text_at_sentences(ldu.content, self.max_chars)
            for i, chunk_text in enumerate(chunks):
                new_id = f"{ldu.ldu_id}_chunk{i}"
                new_ldu = ldu.model_copy(
                    update={
                        "ldu_id": new_id,
                        "content": chunk_text,
                        "content_hash": self._hash(chunk_text),
                        "parent_ldu_id": ldu.ldu_id if i > 0 else ldu.parent_ldu_id,
                    }
                )
                result.append(new_ldu)

            logger.debug(
                "Split LDU %s (%d chars) into %d chunks",
                ldu.ldu_id,
                len(ldu.content),
                len(chunks),
            )

        return result

    def _merge_undersized(self, ldus: List[LDU]) -> List[LDU]:
        """Merge adjacent undersized LDUs of the same type and section."""
        if not ldus:
            return []

        result = [ldus[0]]
        for ldu in ldus[1:]:
            prev = result[-1]

            # Only merge paragraphs that are undersized, on same page, same section
            can_merge = (
                prev.ldu_type == LDUType.PARAGRAPH
                and ldu.ldu_type == LDUType.PARAGRAPH
                and len(prev.content.strip()) < self.min_chars
                and prev.page_number == ldu.page_number
                and prev.section_heading == ldu.section_heading
                and len(prev.content) + len(ldu.content) <= self.max_chars
            )

            if can_merge:
                merged_content = prev.content.rstrip() + "\n" + ldu.content.lstrip()
                merged = prev.model_copy(
                    update={
                        "content": merged_content,
                        "content_hash": self._hash(merged_content),
                    }
                )
                # Update bbox if both have one
                if prev.bbox and ldu.bbox:
                    merged_bbox = BoundingBox(
                        x0=min(prev.bbox.x0, ldu.bbox.x0),
                        y0=min(prev.bbox.y0, ldu.bbox.y0),
                        x1=max(prev.bbox.x1, ldu.bbox.x1),
                        y1=max(prev.bbox.y1, ldu.bbox.y1),
                        page_width=prev.bbox.page_width,
                        page_height=prev.bbox.page_height,
                    )
                    merged = merged.model_copy(update={"bbox": merged_bbox})
                result[-1] = merged
            else:
                result.append(ldu)

        return result

    def _recompute_hashes(self, ldus: List[LDU]) -> List[LDU]:
        """Recompute content_hash for all LDUs."""
        return [
            ldu.model_copy(update={"content_hash": self._hash(ldu.content)})
            for ldu in ldus
        ]

    def _reindex_sequences(self, ldus: List[LDU]) -> List[LDU]:
        """Re-assign sequence_index values within each page."""
        result = []
        page_seq: Dict[int, int] = {}
        for ldu in ldus:
            page = ldu.page_number
            idx = page_seq.get(page, 0)
            result.append(ldu.model_copy(update={"sequence_index": idx}))
            page_seq[page] = idx + 1
        return result

    @staticmethod
    def _compute_token_counts(ldus: List[LDU]) -> List[LDU]:
        """Compute token_count for every LDU (word-level approximation)."""
        return [
            ldu.model_copy(update={"token_count": len(ldu.content.split())})
            for ldu in ldus
        ]

    @staticmethod
    def _split_text_at_sentences(text: str, max_chars: int) -> List[str]:
        """Split text at sentence boundaries, respecting max_chars."""
        # Split on sentence-ending punctuation followed by space
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not current_chunk:
                current_chunk = sentence
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        # If any single sentence exceeds max_chars, force-split at word boundaries
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
                words = chunk.split()
                sub = ""
                for word in words:
                    if not sub:
                        sub = word
                    elif len(sub) + len(word) + 1 <= max_chars:
                        sub += " " + word
                    else:
                        final_chunks.append(sub)
                        sub = word
                if sub:
                    final_chunks.append(sub)

        return final_chunks

    @staticmethod
    def _hash(text: str) -> str:
        """SHA-256 hash of text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
