#!/usr/bin/env python3
"""Search processed documents using semantic search.

Usage:
    python scripts/search_documents.py "total assets"
    python scripts/search_documents.py "revenue growth" --top-k 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.query_tools import VectorStore, semantic_search


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/search_documents.py <query> [--top-k N]")
        sys.exit(1)

    top_k = 5
    args = list(sys.argv[1:])
    if "--top-k" in args:
        idx = args.index("--top-k")
        top_k = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    query = " ".join(args)

    print(f"Searching for: '{query}' (top {top_k})")
    print(f"{'='*60}")

    store = VectorStore()
    results = semantic_search(store, query, top_k=top_k)

    if not results:
        print("No results found. Have you ingested documents?")
        return

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"\n[{i}] {r.get('ldu_id', 'unknown')}")
        print(f"    Page:       {meta.get('page_number', '?')}")
        print(f"    Type:       {meta.get('ldu_type', '?')}")
        print(f"    Section:    {meta.get('section_heading', 'N/A')}")
        print(f"    Confidence: {meta.get('confidence', 0):.3f}")
        print(f"    Distance:   {r.get('distance', 0):.4f}")
        print(f"    Content:    {r.get('content', '')[:200]}...")


if __name__ == "__main__":
    main()
