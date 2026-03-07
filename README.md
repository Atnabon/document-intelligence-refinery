# Document Intelligence Refinery

**Agentic Pipeline for Unstructured Document Extraction at Enterprise Scale**

## Overview

The Document Intelligence Refinery is a classification-aware, spatially-indexed, provenance-preserving extraction system that processes heterogeneous PDF documents into structured, queryable data. It handles four document classes:

| Class | Type | Example |
|-------|------|---------|
| A | Annual Financial Report (native digital) | CBE Annual Report 2023-24 |
| B | Scanned Government/Legal Document (image-based) | DBE Audit Report 2023 |
| C | Technical Assessment Report (mixed layout) | FTA Performance Survey 2022 |
| D | Structured Data Report (table-heavy) | Tax Expenditure Ethiopia 2021-22 |

## Architecture

The pipeline consists of 5 stages:

```
PDF Input → [1. Triage] → [2. Extraction] → [3. Chunking] → [4. Indexing] → [5. Query]
                │               │
                ▼               ▼
         DocumentProfile   LDUs + Ledger
```

### Stage 1: Triage Agent
- Detects origin type (native digital / scanned / mixed)
- Assesses layout complexity (simple / moderate / complex)
- Classifies domain (financial / legal / technical / structured data)
- Recommends extraction strategy

### Stage 2: Extraction Router
- Routes to Strategy A (fast text), B (layout-aware), or C (VLM)
- Confidence-gated escalation: auto-upgrades if quality is low
- Cost budget guard prevents runaway API costs

### Stage 3: Chunking Engine
- Splits LDUs into semantically coherent chunks
- 5 validation rules: table integrity, caption binding, section coherence, minimum context, maximum size
- Config-driven via `extraction_rules.yaml` (min/max chars, overlap)

### Stage 4: PageIndex Builder
- Constructs a hierarchical section tree from document headings
- Infers heading levels via numbered patterns, ALL CAPS, and title-case heuristics
- LLM-powered section summaries with heuristic fallback

### Stage 5: Query Agent
- 3 retrieval tools: `pageindex_navigate`, `semantic_search`, `structured_query`
- Vector store (ChromaDB) with keyword fallback
- FactTable (SQLite) for numeric queries with unit-aware parsing
- Audit mode: verifies claims against source LDUs with provenance chains

## Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd document-intelligence-refinery

# Install with uv (creates .venv automatically)
uv sync --extra dev

# Or install all optional dependencies (dev + google + rag)
uv sync --all-extras
```

### Configuration

Copy your PDF corpus into the `data/` directory, or point the pipeline at your data folder:

```bash
# Set API key for VLM strategy (Strategy C)
export OPENAI_API_KEY="your-key-here"

# Optional: use Google Gemini instead
export VLM_PROVIDER="google"
export GOOGLE_API_KEY="your-key-here"
```

### Running the Pipeline

```bash
# Process a single document
uv run python main.py --input data/CBE\ ANNUAL\ REPORT\ 2023-24.pdf

# Process entire corpus
uv run python main.py --input data/

# Process with max document limit
uv run python main.py --input data/ --max-docs 12

# Custom output directory
uv run python main.py --input data/ --output .refinery/

# Interactive query mode (after processing)
uv run python main.py --input data/report.pdf --query "What was total revenue in 2023?"

# Verify a claim against source documents
uv run python main.py --input data/report.pdf --verify "Revenue increased by 15%"
```

### Output Structure

```
.refinery/
├── profiles/                    # DocumentProfile JSON per document
│   ├── cbe_annual_report_2023_24.json
│   └── ...
├── extraction_ledger.jsonl      # One JSON line per processed document
├── pageindex/                   # PageIndex trees (hierarchical JSON)
│   ├── cbe_annual_report_2023_24_pageindex.json
│   └── ...
├── chunks/                      # Validated semantic chunks
└── qa_examples/                 # Q&A examples with provenance
```

### Running Tests

```bash
uv run pytest tests/ -v
```

## Extraction Rules

All thresholds, chunking rules, and strategy routing logic are externalized in [`rubric/extraction_rules.yaml`](rubric/extraction_rules.yaml). To onboard a new document type, modify this file — not the source code.

## Cost Model

| Strategy | Cost/Page | Speed | Use Case |
|----------|-----------|-------|----------|
| A: Fast Text | $0.0001 | 100+ pg/s | Simple native digital |
| B: Layout-Aware | $0.001 | ~10 pg/s | Tables, multi-column |
| C: Vision Model | $0.01 | ~1 pg/s | Scanned, complex visual |

## Project Structure

```
document-intelligence-refinery/
├── main.py                          # Pipeline entry point
├── pyproject.toml                   # Dependencies & project config
├── README.md                        # This file
├── Report.md                        # Final submission report
├── rubric/
│   └── extraction_rules.yaml       # Externalized configuration
├── src/
│   ├── models/                      # Pydantic schemas
│   │   ├── document_profile.py      # DocumentProfile
│   │   ├── extracted_document.py    # ExtractedDocument + LedgerEntry
│   │   ├── ldu.py                   # Logical Document Unit
│   │   ├── page_index.py           # PageIndex + PageNode
│   │   └── provenance.py           # ProvenanceChain + BoundingBox
│   ├── agents/
│   │   ├── triage.py               # Triage Agent (Phase 1)
│   │   ├── extractor.py            # ExtractionRouter (Phase 2)
│   │   ├── chunker.py              # ChunkingEngine (Phase 3)
│   │   ├── indexer.py              # PageIndexBuilder (Phase 4)
│   │   └── query_agent.py          # QueryAgent (Phase 5)
│   ├── strategies/
│   │   ├── base.py                 # BaseExtractor interface
│   │   ├── fast_text.py            # Strategy A: FastTextExtractor
│   │   ├── layout_extractor.py     # Strategy B: LayoutExtractor
│   │   └── vision_extractor.py     # Strategy C: VisionExtractor
│   └── tools/
│       ├── __init__.py
│       └── query_tools.py          # VectorStore, FactTable, AuditMode
├── scripts/
│   ├── generate_profiles.py         # Batch profile generation
│   └── generate_final_artifacts.py  # PageIndex + Q&A artifact generation
├── tests/                           # Unit tests (54 tests)
├── data/                            # PDF corpus (not committed)
├── .refinery/                       # Pipeline outputs
│   ├── profiles/
│   ├── extraction_ledger.jsonl
│   └── pageindex/
└── docs/
    └── VIDEO_DEMO_SCRIPT.md         # 5-minute demo script
```

## License

MIT
