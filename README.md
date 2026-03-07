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
- Detects origin type (native digital / scanned / mixed) via character density signals
- Assesses layout complexity (simple / moderate / complex)
- Classifies domain (financial / legal / technical / structured data)
- Recommends extraction strategy — all thresholds in `rubric/extraction_rules.yaml`

### Stage 2: Extraction Router
- Routes to Strategy A (fast text), B (Docling layout-aware), or C (Ollama VLM)
- Confidence-gated escalation: auto-upgrades if confidence drops below 0.70
- Cost budget guard prevents runaway API costs

### Stage 3: Chunking Engine
- Splits LDUs into semantically coherent chunks (Logical Document Units)
- 5 validation rules: table integrity, caption binding, section coherence, minimum context, maximum size
- Config-driven via `extraction_rules.yaml` (min/max chars, overlap)

### Stage 4: PageIndex Builder
- Constructs a hierarchical section tree from document headings
- Infers heading levels via numbered patterns, ALL CAPS, and title-case heuristics
- LLM-powered section summaries with heuristic fallback
- NER-enriched metadata: monetary amounts, percentages, years, domain acronyms

### Stage 5: Query Agent
- 3 retrieval tools: `pageindex_navigate`, `semantic_search`, `structured_query`
- Vector store (ChromaDB) with keyword fallback
- FactTable (SQLite) for numeric queries with unit-aware parsing
- Audit mode: verifies claims against source LDUs with full provenance chains

## Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.ai/) (for Strategy C OCR/VLM — free, runs locally)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd document-intelligence-refinery

# Install core dependencies
uv sync

# Install with dev extras (testing, linting)
uv sync --extra dev

# Install with Docling support (deep layout extraction for Strategy B)
uv sync --extra docling

# Install all optional extras
uv sync --all-extras
```

### Ollama Setup (for Strategy C — Vision/OCR)

Strategy C handles scanned documents using [Ollama](https://ollama.ai/) with
`kimi-k2.5:cloud` — a local VLM with no API key required.

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the vision model
ollama pull kimi-k2.5:cloud

# Start the Ollama server (runs on localhost:11434)
ollama serve
```

The pipeline uses Ollama VLM for scanned document extraction (Strategy C),
and Docling for layout-aware extraction (Strategy B). Both run locally.
No `OPENAI_API_KEY` or `GOOGLE_API_KEY` is required.

> **To use OpenAI GPT-4o instead**: set `VLM_PROVIDER=openai` and `OPENAI_API_KEY=<key>`.

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

### Using the Script Runners

Individual pipeline stages can be invoked directly:

```bash
# Stage 1: Triage a document
uv run python scripts/run_triage.py data/report.pdf

# Stage 2: Extract content (triage + extraction)
uv run python scripts/run_extraction.py data/report.pdf

# Full ingest: triage + extraction + chunking + indexing
uv run python scripts/ingest_document.py data/report.pdf

# Stage 5: Query the PageIndex
uv run python scripts/run_query.py "What was net interest income in 2024?"

# Semantic search across all indexed documents
uv run python scripts/search_documents.py "capital adequacy ratio"
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

59 tests across triage, extraction, chunking, indexing, query, and provenance.

## Extraction Rules

All thresholds, chunking rules, and strategy routing logic are externalized in
[`rubric/extraction_rules.yaml`](rubric/extraction_rules.yaml). To onboard a new
document type, modify this file — not the source code.

## Cost Model

| Strategy | Cost/Page | Speed | Use Case |
|----------|-----------|-------|----------|
| A: Fast Text | $0.0001 | 100+ pg/s | Simple native digital |
| B: Docling Layout | $0.001 | ~10 pg/s | Tables, multi-column |
| C: Ollama VLM | $0.000 | ~2 pg/s | Scanned/complex (local, free) |
| C: OpenAI GPT-4o | $0.010 | ~1 pg/s | Cloud fallback (paid, optional) |

Strategy B uses Docling (IBM) for deep layout understanding with a
DoclingDocumentAdapter that normalizes output to the internal LDU schema.
When Docling is not installed, it falls back to pdfplumber+PyMuPDF.
Strategy C uses Ollama `kimi-k2.5:cloud` for scanned documents —
runs entirely on local hardware, free.

## Project Structure

```
document-intelligence-refinery/
├── main.py                          # Pipeline entry point
├── pyproject.toml                   # Dependencies & project config
├── README.md                        # This file
├── Report.md                        # Final submission report
├── DOMAIN_NOTES.md                  # Domain analysis for 4 document classes
├── rubric/
│   └── extraction_rules.yaml       # Externalized configuration (all thresholds)
├── src/
│   ├── models/                      # Pydantic schemas
│   │   ├── document_profile.py      # DocumentProfile + ExtractionCost enum
│   │   ├── extracted_document.py    # ExtractedDocument + LedgerEntry
│   │   ├── ldu.py                   # Logical Document Unit (LDU)
│   │   ├── page_index.py           # PageIndex + PageNode + EntityType + DataType
│   │   └── provenance.py           # ProvenanceChain + BoundingBox + AuditRecord
│   ├── agents/
│   │   ├── triage.py               # Triage Agent (Stage 1)
│   │   ├── extractor.py            # ExtractionRouter (Stage 2)
│   │   ├── chunker.py              # ChunkingEngine (Stage 3)
│   │   ├── indexer.py              # PageIndexBuilder (Stage 4)
│   │   └── query_agent.py          # QueryAgent (Stage 5)
│   ├── strategies/
│   │   ├── base.py                 # BaseExtractor interface
│   │   ├── fast_text.py            # Strategy A: FastTextExtractor
│   │   ├── layout_extractor.py     # Strategy B: LayoutExtractor (Docling + pdfplumber fallback)
│   │   └── vision_extractor.py     # Strategy C: Ollama kimi-k2.5:cloud VLM
│   ├── utils/
│   │   ├── hashing.py              # SHA-256 chunk/document hashing utilities
│   │   ├── pdf_utils.py            # PDF analysis helpers (pdfplumber-based)
│   │   ├── confidence.py           # Confidence scoring utilities
│   │   └── budget_guard.py         # BudgetGuard: per-doc/day/month cost limits
│   └── tools/
│       ├── __init__.py
│       └── query_tools.py          # VectorStore, FactTable, AuditMode
├── scripts/
│   ├── generate_profiles.py         # Batch profile generation
│   ├── run_triage.py               # Stage 1 runner
│   ├── run_extraction.py           # Stage 1+2 runner
│   ├── ingest_document.py          # Full ingest runner (stages 1-4)
│   ├── run_query.py                # Query runner
│   └── search_documents.py         # Semantic search runner
├── tests/                           # 59 unit tests
├── data/                            # PDF corpus (not committed)
├── .refinery/                       # Pipeline outputs (git-ignored)
└── docs/
    └── VIDEO_DEMO_SCRIPT.md         # 5-minute demo script
```

## License

MIT
