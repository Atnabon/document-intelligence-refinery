#!/usr/bin/env python3
"""Run extraction on a single PDF — produces LDUs and ledger entry.

Requires a profile to have been generated first (via run_triage.py).

Usage:
    python scripts/run_extraction.py data/raw/CBE_ANNUAL_REPORT_2023-24.pdf
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_extraction.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    # Phase 1: Triage
    print(f"Triaging: {pdf_path.name}")
    agent = TriageAgent()
    profile = agent.profile_document(pdf_path)
    print(f"  Strategy: {profile.recommended_strategy}")

    # Phase 2: Extract
    print(f"Extracting: {pdf_path.name}")
    router = ExtractionRouter()
    result = router.extract_document(profile, pdf_path)

    print(f"  LDUs:       {len(result.ldus)}")
    print(f"  Tables:     {result.ledger_entry.table_count}")
    print(f"  Confidence: {result.metrics.average_confidence:.3f}")
    print(f"  Cost:       ${result.metrics.total_cost_usd:.6f}")
    print(f"  Time:       {result.metrics.extraction_time_seconds:.2f}s")
    if result.metrics.escalation_count > 0:
        print(f"  Escalations: {result.metrics.escalation_count}")
    if result.ledger_entry.needs_human_review:
        print("  ⚠️  Flagged for human review")

    # Save ledger
    ledger_path = Path(".refinery/extraction_ledger.jsonl")
    router.append_to_ledger(result.ledger_entry, ledger_path)
    print(f"  Ledger:     {ledger_path}")


if __name__ == "__main__":
    main()
