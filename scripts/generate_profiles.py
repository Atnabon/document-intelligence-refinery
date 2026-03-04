"""Generate DocumentProfile JSON outputs for corpus documents.

Runs the Triage Agent against a selection of corpus documents (minimum 3 per
class) and saves profiles + ledger entries to .refinery/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.triage import TriageAgent


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / ".refinery"

# Select at least 3 documents per class (4 classes × 3 = 12 minimum)
SELECTED_DOCS = {
    # Class A: Annual Financial Reports (native digital)
    "class_a": [
        "CBE ANNUAL REPORT 2023-24.pdf",
        "Annual_Report_JUNE-2023.pdf",
        "Annual_Report_JUNE-2022.pdf",
    ],
    # Class B: Scanned Government/Legal Documents (image-based)
    "class_b": [
        "Audit Report - 2023.pdf",
        "2018_Audited_Financial_Statement_Report.pdf",
        "2019_Audited_Financial_Statement_Report.pdf",
    ],
    # Class C: Technical Assessment Reports (mixed)
    "class_c": [
        "fta_performance_survey_final_report_2022.pdf",
        "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
        "Security_Vulnerability_Disclosure_Standard_Procedure_1.pdf",
    ],
    # Class D: Structured Data Reports (table-heavy)
    "class_d": [
        "tax_expenditure_ethiopia_2021_22.pdf",
        "Consumer Price Index August 2025.pdf",
        "Consumer Price Index July 2025.pdf",
    ],
}


def main():
    agent = TriageAgent()
    profiles_dir = OUTPUT_DIR / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    ledger_lines = []
    total = 0
    errors = 0

    for doc_class, filenames in SELECTED_DOCS.items():
        print(f"\n{'='*60}")
        print(f"Processing {doc_class.upper()}")
        print(f"{'='*60}")

        for filename in filenames:
            pdf_path = DATA_DIR / filename
            if not pdf_path.exists():
                print(f"  SKIP: {filename} not found")
                errors += 1
                continue

            try:
                profile = agent.profile_document(pdf_path)
                
                # Save profile JSON
                profile_path = profiles_dir / f"{profile.document_id}.json"
                with open(profile_path, "w") as f:
                    f.write(profile.model_dump_json(indent=2))

                # Build ledger line
                ledger = {
                    "document_id": profile.document_id,
                    "filename": profile.filename,
                    "strategy_selected": profile.recommended_strategy,
                    "confidence_score": 0.85,  # pre-extraction estimate
                    "cost_estimate_usd": profile.estimated_cost_usd,
                    "origin_type": profile.origin_type,
                    "layout_complexity": profile.layout_complexity,
                    "domain_hint": profile.domain_hint,
                    "page_count": profile.page_count,
                    "has_tables": profile.has_tables,
                    "scanned_page_ratio": profile.scanned_page_ratio,
                }
                ledger_lines.append(json.dumps(ledger))

                print(f"  OK: {filename}")
                print(f"      → origin={profile.origin_type}, complexity={profile.layout_complexity}")
                print(f"      → domain={profile.domain_hint}, strategy={profile.recommended_strategy}")
                print(f"      → pages={profile.page_count}, cost=${profile.estimated_cost_usd:.4f}")
                total += 1

            except Exception as e:
                print(f"  ERROR: {filename}: {e}")
                errors += 1

    # Write ledger
    ledger_path = OUTPUT_DIR / "extraction_ledger.jsonl"
    with open(ledger_path, "w") as f:
        f.write("\n".join(ledger_lines) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE: {total} documents profiled, {errors} errors")
    print(f"Profiles: {profiles_dir}")
    print(f"Ledger:   {ledger_path}")


if __name__ == "__main__":
    main()
