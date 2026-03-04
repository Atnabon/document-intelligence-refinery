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

### Stage 3–5: (Final submission)
- Semantic chunking, PageIndex tree building, and query agent

## Quick Start

### Prerequisites
- Python 3.10+
- pip or uv

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd document-intelligence-refinery

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
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
python main.py --input data/CBE\ ANNUAL\ REPORT\ 2023-24.pdf

# Process entire corpus
python main.py --input data/

# Process with max document limit
python main.py --input data/ --max-docs 12

# Custom output directory
python main.py --input data/ --output .refinery/
```

### Output Structure

```
.refinery/
├── profiles/                    # DocumentProfile JSON per document
│   ├── cbe_annual_report_2023_24.json
│   └── ...
├── extraction_ledger.jsonl      # One JSON line per processed document
└── pageindex/                   # PageIndex trees (final submission)
```

### Running Tests

```bash
pytest tests/ -v
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
├── Report.md                        # Interim submission report
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
│   │   └── extractor.py            # ExtractionRouter (Phase 2)
│   ├── strategies/
│   │   ├── base.py                 # BaseExtractor interface
│   │   ├── fast_text.py            # Strategy A: FastTextExtractor
│   │   ├── layout_extractor.py     # Strategy B: LayoutExtractor
│   │   └── vision_extractor.py     # Strategy C: VisionExtractor
│   └── tools/                      # Utility functions
├── tests/                           # Unit tests
├── data/                            # PDF corpus (not committed)
├── .refinery/                       # Pipeline outputs
│   ├── profiles/
│   ├── extraction_ledger.jsonl
│   └── pageindex/
└── docs/                            # Additional documentation
```

## License

MIT
