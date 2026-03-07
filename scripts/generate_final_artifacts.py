"""Generate PageIndex artifacts and Q&A examples for final submission.

This script:
1. Processes 12 documents (3 per class) through the full pipeline
2. Generates PageIndex JSON for each document
3. Runs 3 sample Q&A queries per document class (12 total)
4. Outputs all artifacts to .refinery/pageindex/ and .refinery/qa_examples/

Usage:
    python scripts/generate_final_artifacts.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.tools.query_tools import FactTable, VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_artifacts")


# ── Document Selection: 3 per class (12 total) ───────────────────────────

CORPUS = {
    "A": [  # Financial Reports (native digital)
        "CBE Annual Report 2023-24.pdf",
        "Annual_Report_JUNE-2023.pdf",
        "Annual_Report_JUNE-2022.pdf",
    ],
    "B": [  # Scanned Government/Legal (image-based)
        "Audit Report - 2023.pdf",
        "2018_Audited_Financial_Statement_Report.pdf",
        "2019_Audited_Financial_Statement_Report.pdf",
    ],
    "C": [  # Technical Assessment Reports (mixed)
        "fta_performance_survey_final_report_2022.pdf",
        "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
        "Security_Vulnerability_Disclosure_Standard_Procedure_1.pdf",
    ],
    "D": [  # Structured Data Reports (table-heavy)
        "tax_expenditure_ethiopia_2021_22.pdf",
        "Consumer Price Index August 2025.pdf",
        "Consumer Price Index July 2025.pdf",
    ],
}

# ── Q&A Examples: 3 questions per class ───────────────────────────────────

QA_TEMPLATES = {
    "A": [
        "What is the total assets figure reported for the fiscal year?",
        "What was the net income or profit for the reporting period?",
        "How many branches does the bank operate?",
    ],
    "B": [
        "What is the auditor's opinion on the financial statements?",
        "What is the total revenue or income reported?",
        "What period do the audited financial statements cover?",
    ],
    "C": [
        "What are the key findings or recommendations of the assessment?",
        "What methodology was used for the evaluation?",
        "What is the overall performance rating or score?",
    ],
    "D": [
        "What is the total tax expenditure for the most recent fiscal year?",
        "Which category has the highest import tax expenditure?",
        "What is the consumer price index value for the reporting period?",
    ],
}


def load_config() -> dict:
    """Load pipeline configuration."""
    import yaml

    config_path = Path("rubric/extraction_rules.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    config = load_config()
    data_dir = Path("data")
    output_dir = Path(".refinery")

    # Initialize agents
    triage_agent = TriageAgent(config=config)
    router = ExtractionRouter(config=config)
    chunker = ChunkingEngine(config=config)
    indexer = PageIndexBuilder(config=config)
    query_agent = QueryAgent(config=config)

    # Create output directories
    pageindex_dir = output_dir / "pageindex"
    qa_dir = output_dir / "qa_examples"
    pageindex_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    all_qa_results = []
    processed_docs = {}

    for doc_class, filenames in CORPUS.items():
        logger.info("\n" + "=" * 70)
        logger.info("Processing Class %s documents", doc_class)
        logger.info("=" * 70)

        for filename in filenames:
            pdf_path = data_dir / filename
            if not pdf_path.exists():
                # Try to find with different naming
                matches = list(data_dir.glob(f"*{filename.split('.')[0]}*"))
                if matches:
                    pdf_path = matches[0]
                else:
                    logger.warning("File not found: %s — skipping", filename)
                    continue

            try:
                logger.info("Processing: %s (Class %s)", pdf_path.name, doc_class)

                # Phase 1: Triage
                profile = triage_agent.profile_document(pdf_path)

                # Save profile
                profiles_dir = output_dir / "profiles"
                profiles_dir.mkdir(parents=True, exist_ok=True)
                profile_path = profiles_dir / f"{profile.document_id}.json"
                with open(profile_path, "w") as f:
                    f.write(profile.model_dump_json(indent=2))

                # Phase 2: Extraction
                result = router.extract_document(profile, pdf_path)

                # Save ledger
                ledger_path = output_dir / "extraction_ledger.jsonl"
                router.append_to_ledger(result.ledger_entry, ledger_path)

                # Phase 3: Chunking
                chunked_ldus = chunker.chunk(result.ldus)
                result.ldus = chunked_ldus

                # Phase 4: PageIndex
                page_index = indexer.build(
                    document_id=profile.document_id,
                    ldus=chunked_ldus,
                    document_title=profile.filename,
                    generate_summaries=True,
                )
                result.page_index = page_index
                indexer.save(page_index)

                logger.info(
                    "  PageIndex: %d sections, depth %d",
                    page_index.total_sections,
                    page_index.max_depth,
                )

                # Phase 5: Register with Query Agent
                query_agent.register_document(result, page_index)
                processed_docs[profile.document_id] = {
                    "class": doc_class,
                    "filename": pdf_path.name,
                }

            except Exception as e:
                logger.error("Failed to process %s: %s", filename, str(e))
                import traceback
                traceback.print_exc()
                continue

    # ── Generate Q&A Examples ─────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Generating Q&A Examples")
    logger.info("=" * 70)

    for doc_id, doc_info in processed_docs.items():
        doc_class = doc_info["class"]
        questions = QA_TEMPLATES.get(doc_class, [])

        for q_idx, question in enumerate(questions):
            try:
                logger.info(
                    "Q&A [Class %s][%s]: %s",
                    doc_class,
                    doc_info["filename"],
                    question[:60],
                )
                result = query_agent.get_qa_example(question, doc_id)
                result["document_filename"] = doc_info["filename"]
                all_qa_results.append(result)

                # Log answer preview
                answer = result.get("answer", "")
                logger.info("  Answer: %s...", answer[:100])

            except Exception as e:
                logger.error("Q&A failed for %s: %s", doc_id, str(e))

    # Save Q&A examples
    qa_path = qa_dir / "qa_examples.json"
    with open(qa_path, "w") as f:
        json.dump(all_qa_results, f, indent=2, default=str)
    logger.info("Saved %d Q&A examples to %s", len(all_qa_results), qa_path)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Final Artifact Generation Summary")
    logger.info("=" * 70)
    logger.info("Documents processed: %d / 12", len(processed_docs))
    logger.info("PageIndex files: %d", len(list(pageindex_dir.glob("*.json"))))
    logger.info("Q&A examples: %d / 12", len(all_qa_results))


if __name__ == "__main__":
    main()
