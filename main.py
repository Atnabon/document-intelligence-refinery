"""Main pipeline entry point for the Document Intelligence Refinery.

Usage:
    python main.py --input <pdf_path_or_directory>
    python main.py --input data/ --output .refinery/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("refinery")


def load_config(config_path: Path | None = None) -> dict:
    """Load pipeline configuration from extraction_rules.yaml."""
    if config_path and config_path.exists():
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def process_document(
    pdf_path: Path,
    output_dir: Path,
    triage_agent: TriageAgent,
    router: ExtractionRouter,
) -> None:
    """Process a single PDF through the full pipeline."""
    logger.info("=" * 60)
    logger.info("Processing: %s", pdf_path.name)
    logger.info("=" * 60)

    # ── Phase 1: Triage ───────────────────────────────────────────────
    profile = triage_agent.profile_document(pdf_path)

    # Save profile
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_dir / f"{profile.document_id}.json"
    with open(profile_path, "w") as f:
        f.write(profile.model_dump_json(indent=2))
    logger.info("Profile saved: %s", profile_path)

    # ── Phase 2: Extraction ───────────────────────────────────────────
    result = router.extract_document(profile, pdf_path)

    # Save ledger entry
    ledger_path = output_dir / "extraction_ledger.jsonl"
    router.append_to_ledger(result.ledger_entry, ledger_path)

    # Print summary
    logger.info("Strategy: %s", result.metrics.strategy_used)
    logger.info("LDUs extracted: %d", len(result.ldus))
    logger.info("Tables found: %d", result.ledger_entry.table_count)
    logger.info("Confidence: %.3f", result.metrics.average_confidence)
    logger.info("Cost: $%.6f", result.metrics.total_cost_usd)
    logger.info("Time: %.2fs", result.metrics.extraction_time_seconds)
    if result.metrics.escalation_count > 0:
        logger.info("Escalations: %d", result.metrics.escalation_count)
    if result.ledger_entry.needs_human_review:
        logger.warning(
            "HUMAN REVIEW REQUIRED: %s — confidence %.3f below threshold",
            pdf_path.name,
            result.metrics.average_confidence,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Document Intelligence Refinery — Process PDFs into structured data."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to a PDF file or directory of PDFs.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".refinery",
        help="Output directory for profiles and ledger (default: .refinery).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="rubric/extraction_rules.yaml",
        help="Path to extraction rules config.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config)

    # Load config
    config = load_config(config_path)

    # Initialize agents
    triage_agent = TriageAgent(config=config)
    router = ExtractionRouter(config=config)

    # Collect PDF files
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
    else:
        logger.error("Invalid input: %s (must be a PDF file or directory)", input_path)
        sys.exit(1)

    if args.max_docs:
        pdf_files = pdf_files[: args.max_docs]

    logger.info("Found %d PDF files to process.", len(pdf_files))

    # Process each document
    for pdf_file in pdf_files:
        try:
            process_document(pdf_file, output_dir, triage_agent, router)
        except Exception as e:
            logger.error("Failed to process %s: %s", pdf_file.name, str(e))
            continue

    logger.info("Pipeline complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
