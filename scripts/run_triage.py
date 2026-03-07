#!/usr/bin/env python3
"""Run triage on a single PDF — produces a DocumentProfile JSON.

Usage:
    python scripts/run_triage.py data/raw/CBE_ANNUAL_REPORT_2023-24.pdf
    python scripts/run_triage.py data/raw/*.pdf
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.triage import TriageAgent


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_triage.py <pdf_path> [pdf_path ...]")
        sys.exit(1)

    agent = TriageAgent()

    for pdf_path_str in sys.argv[1:]:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Triaging: {pdf_path.name}")
        print(f"{'='*60}")

        profile = agent.profile_document(pdf_path)

        # Save profile
        out_dir = Path(".refinery/profiles")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{profile.document_id}.json"
        with open(out_path, "w") as f:
            f.write(profile.model_dump_json(indent=2))

        print(f"  Origin:     {profile.origin_type}")
        print(f"  Layout:     {profile.layout_complexity}")
        print(f"  Domain:     {profile.domain_hint}")
        print(f"  Strategy:   {profile.recommended_strategy}")
        print(f"  Pages:      {profile.page_count}")
        print(f"  Saved:      {out_path}")


if __name__ == "__main__":
    main()
