# Domain Notes — Document Intelligence Refinery

## Overview

This document captures domain-specific decisions, failure modes, and design trade-offs
for each document class handled by the pipeline. It serves as the engineering companion
to `Report.md` and the `rubric/extraction_rules.yaml` configuration.

---

## Document Classes

### Class A — Native-Digital Financial Reports

**Characteristics:**
- Born-digital PDFs (character stream embedded)
- Heavy use of tables: balance sheets, income statements, cash-flow statements
- Multi-column layouts common in annual reports
- Embedded fonts, consistent styling
- Page counts: 50-300+ pages

**Optimal Strategy:** Strategy B (Layout-Aware via Docling)

**Why not Strategy A?** Fast text extraction mangles tables — column alignment is lost,
merged cells become garbled text. For a financial report, table integrity is paramount.

**Why not Strategy C?** Native-digital text is extractable without VLM. Using
vision models on born-digital PDFs wastes compute time with no accuracy gain.
Docling provides deep layout understanding for free, locally.

**Known Failure Modes:**
- **Merged cells in pdfplumber:** Multi-level column headers sometimes cause cell
  duplication. Mitigation: post-processing deduplication in ChunkingEngine.
- **Footnote separation:** Footnotes at page bottom may merge into adjacent table cells.
  Mitigation: spatial Y-coordinate heuristic to separate footer regions.
- **Currency symbol UTF-8:** Ethiopian Birr (ETB) symbol may appear as mojibake in some
  PDF generators. Mitigation: regex normalization in fact extraction.

**Entity Patterns:**
- Monetary: `ETB \d+`, `\$[\d,.]+[BMK]?`
- Percentages: `\d+\.?\d*%`
- Fiscal years: `FY\s*20\d{2}[/-]?\d{2,4}`

---

### Class B — Scanned Government/Legal Audits

**Characteristics:**
- Image-only PDFs (no character stream)
- Low print quality, uneven margins, skewed scans
- Stamps, handwritten annotations, signatures
- Often photocopies of photocopies (generation loss)

**Optimal Strategy:** Strategy C (Vision Model via Ollama)

**Why Ollama VLM?**
- Runs entirely locally via `ollama serve` — no API cost, no API key required
- kimi-k2.5:cloud model provides structured JSON output with table detection
- Handles degraded scans, stamps, and handwritten annotations
- Produces bounding box estimates for provenance

**Why not paid cloud VLM?**
- Scanned audit pages are text-dense; local VLM recovers 85-95% of content
- Cloud VLM adds $0.01/page expense with marginal improvement for text-only pages
- Budget guard limits exist: $2.00/doc, $5.00/doc max
- For 200-page scanned documents, cloud VLM cost = $2.00 vs $0.00 for Ollama

**Known Failure Modes:**
- **Low resolution stamps overlap text:** OCR may merge stamp text into body text.
  Mitigation: confidence-based filtering (stamps produce lower-confidence characters).
- **Handwritten annotations:** OCR accuracy drops below 50% on handwriting.
  Mitigation: flag low-confidence regions for human review.
- **Table detection on scans:** pdfplumber cannot detect tables in image-only pages.
  Mitigation: VisionExtractor includes basic table structure inference from OCR output.

---

### Class C — Mixed Technical Assessments

**Characteristics:**
- Mix of native-digital text and embedded images/charts/maps
- Section-heavy structure with numbered headings
- Charts and graphs that require vision models for data extraction
- Cross-references between sections ("see Section 3.2", "refer to Figure 5")

**Optimal Strategy:** Strategy B (Docling) with selective Strategy C escalation

**Decision Tree:**
1. Start with Strategy B (layout-aware) — handles the text+table majority
2. If page has >50% image coverage and <50 characters → escalate that page to Strategy C
3. Per-page escalation avoids whole-document VLM cost

**Known Failure Modes:**
- **Cross-reference resolution:** References like "see Table 3" need a lookup index.
  Mitigation: ChunkingEngine builds `reference_map` from (type, number) → LDU ID.
- **Chart data extraction:** Chart images require VLM to extract underlying data.
  Mitigation: charts are preserved as figure LDUs with captions; numerical data should
  come from accompanying tables.
- **Section numbering inconsistency:** Some reports use "1.1", others use "I.A".
  Mitigation: PageIndexBuilder uses multi-pattern heading detection (decimal, roman,
  alphabetic).

---

### Class D — Table-Heavy Structured Data Reports

**Characteristics:**
- Dominated by tables (70-90% of content is tabular)
- Relatively simple layouts (single-column with wide tables)
- Statistical data, fiscal summaries, CPI tables
- Headers may span multiple rows

**Optimal Strategy:** Strategy B (Docling Layout-Aware) — specifically the table extraction pipeline

**Key Design Decision:** For Class D, the ChunkingEngine treats every table as a single
LDU (Rule 1: table integrity). This means a 50-row table stays as one chunk even if it
exceeds the typical `max_chars` limit. The rationale: splitting a financial table between
rows destroys its semantic coherence for RAG retrieval.

**Known Failure Modes:**
- **Multi-page tables:** pdfplumber extracts tables per-page, so a table spanning pages
  2-3 appears as two separate tables. Mitigation: chunking engine detects continuation
  patterns (same headers on consecutive pages) and merges them.
- **Sub-header rows in data:** Some tables use sub-header rows (e.g., "Section B: Revenue")
  mixed with data rows. pdfplumber cannot distinguish these from data.
  Mitigation: fact extraction uses column-header context for key generation.
- **Empty cells:** Statistical tables have many empty cells (data not applicable for
  certain periods). Mitigation: `_parse_numeric()` in FactTable handles empty strings
  as None rather than 0.

---

## Strategy Cost Comparison

| Strategy | Cost/Page | Speed     | Best For                    | API Required |
|----------|-----------|-----------|-----------------------------|--------------|
| A: Fast  | $0.0001   | 100pg/s   | Simple digital text         | No           |
| B: Docling| $0.001   | 10pg/s    | Tables + complex layout     | No           |
| C: Ollama VLM| $0.00 | 1-2pg/s   | Scanned/image-heavy docs    | No (local)   |

*Strategy C cost is $0.00 when using Ollama (local). $0.01/page applies only when 
using a cloud VLM provider (GPT-4o/Gemini) as optional override.

---

## Confidence-Gated Escalation Decision Tree

```
Input: DocumentProfile from Triage Agent
  │
  ├── origin = "native_digital"?
  │     ├── YES → layout = "simple"?
  │     │           ├── YES → Strategy A (fast_text)
  │     │           └── NO  → Strategy B (layout_aware)
  │     └── NO
  │
  ├── origin = "scanned_image"?
  │     └── YES → Strategy C (vision model via Ollama VLM)
  │
  └── origin = "mixed"?
        └── YES → Strategy B with per-page C escalation
              │
              └── For each page:
                    confidence < 0.70? → escalate that page to C
```

After initial extraction:
```
avg_confidence >= 0.70? → ACCEPT result
avg_confidence < 0.70?
  ├── next strategy available?
  │     ├── cost within budget? → ESCALATE
  │     └── cost exceeds budget? → ACCEPT with human_review flag
  └── all strategies exhausted? → ACCEPT with human_review flag
```

---

## Key Design Decisions

### 1. Docling for Layout-Aware Extraction (Strategy B)
The pipeline uses Docling (IBM) as the primary layout-aware extraction engine:
- **Docling**: Deep document understanding with structural element detection (tables, figures, headings, lists)
- **DoclingDocumentAdapter**: Normalizes Docling output to the internal LDU schema
- **pdfplumber+PyMuPDF fallback**: Used when Docling is not installed, ensuring graceful degradation
- MinerU was evaluated but Docling was chosen for its cleaner API and MIT-compatible licensing

### 2. Local-First Architecture
All extraction happens locally using:
- **PyMuPDF** for fast text block extraction (Strategy A)
- **Docling** for layout-aware extraction with table structure (Strategy B)
- **pdfplumber** as fallback table detection for Strategy B
- **Ollama kimi-k2.5:cloud** for vision model extraction on scanned pages (Strategy C)
- **SentenceTransformers** (`all-MiniLM-L6-v2`) for local embeddings
- **ChromaDB** for local vector storage
- **SQLite** for structured fact queries

### 3. Five Chunking Rules as Enforceable Constraints
The ChunkingEngine treats the five rules as testable invariants:
1. **Table integrity**: validated — tables never appear in multiple chunks
2. **Caption binding**: validated — captions have `parent_ldu_id` set
3. **List preservation**: validated — lists are single LDUs unless too large
4. **Section propagation**: validated — all non-heading LDUs have `section_heading`
5. **Cross-reference**: advisory — unresolved references logged as warnings

### 4. Provenance-First Query Results
Every answer from the QueryAgent includes a `ProvenanceChain`:
- `document_id` + `page_number` + `bbox` for spatial provenance
- `content_hash` for integrity verification
- `verification_status`: verified / unverified / unverifiable

---

## Ethiopian Document Corpus Notes

The pipeline was designed and tested against Ethiopian government and financial documents:

- **CBE Annual Report 2023-24**: 200+ page native digital report. Dense financial tables,
  Amharic/English bilingual sections. Primary test document for Class A.
- **Audit Report 2023**: Government audit with scanned pages. Stamps, signatures,
  handwritten annotations. Primary test document for Class B.
- **FTA Performance Survey 2022**: Mixed format technical assessment. Charts, numbered
  sections, cross-references. Primary test document for Class C.
- **Tax Expenditure Ethiopia 2021-22**: Table-heavy statistical report. 90% tabular
  content. Primary test document for Class D.

### Currency Handling
Ethiopian Birr (ETB) amounts use comma-separated thousands with no standard
symbol in PDFs. The fact extraction regex handles:
- `ETB 1,234,567.89`
- `Birr 1,234,567`
- `1,234,567 ETB`
- Plain numbers with context (adjacent column header = "Amount (ETB)")
