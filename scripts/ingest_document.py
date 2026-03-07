#!/usr/bin/env python3
"""Run the full ingestion pipeline on a single PDF.

Combines: Triage → Extraction → Chunking → Indexing → Query Registration.

Usage:
    python scripts/ingest_document.py data/raw/CBE_ANNUAL_REPORT_2023-24.pdf
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_document.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    output_dir = Path(".refinery")

    print(f"\n{'='*60}")
    print(f"Ingesting: {pdf_path.name}")
    print(f"{'='*60}")

    # Phase 1: Triage
    print("\n[Phase 1] Triage...")
    triage = TriageAgent()
    profile = triage.profile_document(pdf_path)
    print(f"  Origin: {profile.origin_type}, Layout: {profile.layout_complexity}")
    print(f"  Domain: {profile.domain_hint}, Strategy: {profile.recommended_strategy}")

    # Save profile
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    with open(profiles_dir / f"{profile.document_id}.json", "w") as f:
        f.write(profile.model_dump_json(indent=2))

    # Phase 2: Extraction
    print("\n[Phase 2] Extraction...")
    router = ExtractionRouter()
    result = router.extract_document(profile, pdf_path)
    print(f"  LDUs: {len(result.ldus)}, Tables: {result.ledger_entry.table_count}")
    print(f"  Confidence: {result.metrics.average_confidence:.3f}")

    # Save ledger
    router.append_to_ledger(result.ledger_entry, output_dir / "extraction_ledger.jsonl")

    # Phase 3: Chunking
    print("\n[Phase 3] Semantic Chunking...")
    chunker = ChunkingEngine()
    chunked_ldus = chunker.chunk(result.ldus)
    result.ldus = chunked_ldus
    print(f"  Chunks: {len(chunked_ldus)}")

    # Phase 4: Indexing
    print("\n[Phase 4] PageIndex Building...")
    indexer = PageIndexBuilder()
    page_index = indexer.build(
        document_id=profile.document_id,
        ldus=chunked_ldus,
        document_title=profile.filename,
    )
    result.page_index = page_index
    indexer.save(page_index)
    print(f"  Sections: {page_index.total_sections}, Depth: {page_index.max_depth}")

    # Phase 5: Register with Query Agent
    print("\n[Phase 5] Query Agent Registration...")
    query_agent = QueryAgent()
    query_agent.register_document(result, page_index)
    print("  Document registered for querying")

    print(f"\n{'='*60}")
    print(f"Ingestion complete. Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
