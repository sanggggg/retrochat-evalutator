#!/usr/bin/env python3
"""Standalone script to run both LLM and Clustering Summarizers on an existing raw-rubrics.json file."""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from retrochat_evaluator.config import Config
from retrochat_evaluator.models.rubric import Rubric, RubricList
from retrochat_evaluator.llm.gemini import GeminiClient
from retrochat_evaluator.training.llm_summarizer import RubricSummarizer
from retrochat_evaluator.training.semantic_summarizer import SemanticClusteringSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_raw_rubrics(raw_rubrics_path: Path) -> list[list[Rubric]]:
    """Load raw rubrics from JSON file and convert to list of Rubric lists.
    
    Args:
        raw_rubrics_path: Path to raw-rubrics.json file.
        
    Returns:
        List of rubric lists, one per session.
    """
    with open(raw_rubrics_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    rubric_lists = []
    for session_id, rubric_dicts in raw_data.items():
        rubrics = []
        for rubric_dict in rubric_dicts:
            try:
                rubric = Rubric(**rubric_dict)
                rubrics.append(rubric)
            except Exception as e:
                logger.warning(f"Failed to parse rubric from session {session_id}: {e}")
                continue
        if rubrics:
            rubric_lists.append(rubrics)
    
    logger.info(f"Loaded {len(rubric_lists)} sessions with rubrics from {raw_rubrics_path}")
    total_rubrics = sum(len(r) for r in rubric_lists)
    logger.info(f"Total {total_rubrics} rubrics across all sessions")
    
    return rubric_lists


async def main():
    """Main entry point for the summarizer script."""
    parser = argparse.ArgumentParser(
        description="Run both LLM and Clustering Summarizers on an existing raw-rubrics.json file. "
                    "Outputs rubrics_llm.json and rubrics_cluster.json"
    )
    parser.add_argument(
        "raw_rubrics",
        type=Path,
        help="Path to raw-rubrics.json file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml file (optional, uses defaults if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for rubrics.json (defaults to same directory as raw-rubrics.json)",
    )
    parser.add_argument(
        "--min-rubrics",
        type=int,
        default=None,
        help="Minimum number of rubrics in final list (overrides config)",
    )
    parser.add_argument(
        "--max-rubrics",
        type=int,
        default=None,
        help="Maximum number of rubrics in final list (overrides config)",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Directory containing prompt templates (defaults to prompts/)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model for clustering (overrides config)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Similarity threshold for clustering (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.raw_rubrics.exists():
        logger.error(f"Raw rubrics file not found: {args.raw_rubrics}")
        sys.exit(1)
    
    # Load config
    if args.config and args.config.exists():
        logger.info(f"Loading config from {args.config}")
        config = Config.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = Config()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.raw_rubrics.parent
    
    # Determine prompts directory
    if args.prompts_dir:
        prompts_dir = args.prompts_dir
    else:
        prompts_dir = config.prompts_dir
        # If relative path, resolve relative to project root
        if not prompts_dir.is_absolute():
            prompts_dir = project_root / prompts_dir
    
    if not prompts_dir.exists():
        logger.error(f"Prompts directory not found: {prompts_dir}")
        sys.exit(1)
    
    # Override config values from command line
    min_rubrics = args.min_rubrics if args.min_rubrics is not None else config.training.min_rubrics
    max_rubrics = args.max_rubrics if args.max_rubrics is not None else config.training.max_rubrics
    embedding_model = args.embedding_model if args.embedding_model is not None else config.training.embedding_model
    similarity_threshold = args.similarity_threshold if args.similarity_threshold is not None else config.training.similarity_threshold
    
    logger.info(f"Summarization settings: min_rubrics={min_rubrics}, max_rubrics={max_rubrics}")
    logger.info(f"Clustering settings: embedding_model={embedding_model}, similarity_threshold={similarity_threshold}")
    
    # Load raw rubrics
    rubric_lists = load_raw_rubrics(args.raw_rubrics)
    
    if not rubric_lists:
        logger.error("No rubrics found in raw-rubrics.json file")
        sys.exit(1)
    
    # Initialize LLM client for LLM summarizer
    logger.info(f"Initializing LLM client with model: {config.summarization_llm.model_name}")
    llm_client = GeminiClient(
        config.summarization_llm,
        rate_limiter_config=config.rate_limiter,
    )
    
    # Initialize both summarizers
    logger.info("Initializing LLM summarizer...")
    llm_summarizer = RubricSummarizer(
        llm_client=llm_client,
        prompt_template_path=Path("rubric_summarizer.txt"),
        prompts_dir=prompts_dir,
        min_rubrics=min_rubrics,
        max_rubrics=max_rubrics,
    )
    
    logger.info("Initializing clustering summarizer...")
    cluster_summarizer = SemanticClusteringSummarizer(
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        min_rubrics=min_rubrics,
        max_rubrics=max_rubrics,
        min_samples=max(min_rubrics, 2),
    )
    
    # Run both summarizations in parallel
    logger.info("Starting summarization (both LLM and Clustering)...")
    
    llm_task = asyncio.create_task(llm_summarizer.summarize(rubric_lists))
    cluster_task = asyncio.create_task(cluster_summarizer.summarize(rubric_lists))
    
    # Wait for both to complete
    (llm_rubrics, llm_notes), (cluster_rubrics, cluster_notes) = await asyncio.gather(
        llm_task, cluster_task
    )
    
    # Save LLM summarizer results
    if not llm_rubrics:
        logger.warning("LLM summarization produced no rubrics")
    else:
        logger.info(f"LLM summarization complete. Generated {len(llm_rubrics)} final rubrics")
        llm_rubric_list = RubricList(
            version="1.0",
            created_at=datetime.utcnow(),
            training_config=None,
            rubrics=llm_rubrics,
            consolidation_notes=llm_notes,
        )
        llm_output_path = output_dir / "rubrics_llm.json"
        llm_rubric_list.to_json(llm_output_path)
        logger.info(f"Saved LLM rubrics to {llm_output_path}")
        
        if llm_notes:
            llm_notes_path = output_dir / "consolidation_notes_llm.txt"
            with open(llm_notes_path, "w", encoding="utf-8") as f:
                f.write(llm_notes)
            logger.info(f"Saved LLM consolidation notes to {llm_notes_path}")
    
    # Save Clustering summarizer results
    if not cluster_rubrics:
        logger.warning("Clustering summarization produced no rubrics")
    else:
        logger.info(f"Clustering summarization complete. Generated {len(cluster_rubrics)} final rubrics")
        cluster_rubric_list = RubricList(
            version="1.0",
            created_at=datetime.utcnow(),
            training_config=None,
            rubrics=cluster_rubrics,
            consolidation_notes=cluster_notes,
        )
        cluster_output_path = output_dir / "rubrics_cluster.json"
        cluster_rubric_list.to_json(cluster_output_path)
        logger.info(f"Saved clustering rubrics to {cluster_output_path}")
        
        if cluster_notes:
            cluster_notes_path = output_dir / "consolidation_notes_cluster.txt"
            with open(cluster_notes_path, "w", encoding="utf-8") as f:
                f.write(cluster_notes)
            logger.info(f"Saved clustering consolidation notes to {cluster_notes_path}")
    
    logger.info("Done! Both summarizers have completed.")


if __name__ == "__main__":
    asyncio.run(main())
