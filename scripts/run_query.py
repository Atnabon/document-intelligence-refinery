#!/usr/bin/env python3
"""Run a query against processed documents.

Usage:
    python scripts/run_query.py "What is the total revenue?"
    python scripts/run_query.py --verify "Revenue was $4.2B in Q3"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.query_agent import QueryAgent


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_query.py <question>")
        print("       python scripts/run_query.py --verify <claim>")
        sys.exit(1)

    verify_mode = "--verify" in sys.argv
    if verify_mode:
        sys.argv.remove("--verify")

    question = " ".join(sys.argv[1:])

    agent = QueryAgent()

    if verify_mode:
        print(f"\n{'='*60}")
        print(f"Audit Mode — Claim Verification")
        print(f"Claim: {question}")
        print(f"{'='*60}")

        for doc_id in agent._documents:
            result = agent.verify_claim(question, doc_id)
            print(f"\nDocument: {doc_id}")
            print(f"  Status:     {result['verification_status']}")
            print(f"  Confidence: {result['confidence']:.2f}")
    else:
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}")

        result = agent.query(question)
        print(f"\nAnswer:\n{result['answer']}")

        if result.get("provenance_chain"):
            chain = result["provenance_chain"]
            print(f"\nProvenance:")
            print(f"  Document:   {chain.get('document_name', 'N/A')}")
            print(f"  Confidence: {chain.get('confidence', 0):.2f}")
            for cit in chain.get("citations", []):
                print(f"  - Page {cit.get('page_number', '?')}: {cit.get('text_snippet', '')[:100]}...")


if __name__ == "__main__":
    main()
