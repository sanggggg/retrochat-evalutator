"""Training orchestrator for the complete training pipeline."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..models.rubric import Rubric, RubricList, TrainingConfig as RubricTrainingConfig
from ..config import (
    TrainingConfig,
    ExtractionLLMConfig,
    SummarizationLLMConfig,
    SummarizationMethod,
)
from ..llm.gemini import GeminiClient
from .loader import DatasetLoader
from .extractor import RubricExtractor
from .summarizer import RubricSummarizer
from .semantic_summarizer import SemanticClusteringSummarizer

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrate the complete training pipeline."""

    def __init__(
        self,
        dataset_dir: Path,
        manifest_path: Path,
        prompts_dir: Path,
        config: Optional[TrainingConfig] = None,
        extraction_llm_config: Optional[ExtractionLLMConfig] = None,
        summarization_llm_config: Optional[SummarizationLLMConfig] = None,
    ):
        """Initialize trainer.

        Args:
            dataset_dir: Directory containing JSONL session files.
            manifest_path: Path to the dataset manifest JSON file.
            prompts_dir: Directory containing prompt templates.
            config: Training configuration.
            extraction_llm_config: LLM configuration for rubric extraction.
            summarization_llm_config: LLM configuration for rubric summarization.
        """
        self.config = config or TrainingConfig()
        self.extraction_llm_config = extraction_llm_config or ExtractionLLMConfig()
        self.summarization_llm_config = summarization_llm_config or SummarizationLLMConfig()
        self.prompts_dir = Path(prompts_dir)

        # Initialize components
        self.loader = DatasetLoader(dataset_dir, manifest_path)
        self._extraction_llm_client: Optional[GeminiClient] = None
        self._summarization_llm_client: Optional[GeminiClient] = None
        self._extractor: Optional[RubricExtractor] = None
        self._summarizer: Optional[RubricSummarizer] = None
        self._semantic_summarizer: Optional[SemanticClusteringSummarizer] = None

    def _get_extraction_llm_client(self) -> GeminiClient:
        """Get or create LLM client for extraction."""
        if self._extraction_llm_client is None:
            self._extraction_llm_client = GeminiClient(self.extraction_llm_config)
        return self._extraction_llm_client

    def _get_summarization_llm_client(self) -> GeminiClient:
        """Get or create LLM client for summarization."""
        if self._summarization_llm_client is None:
            self._summarization_llm_client = GeminiClient(self.summarization_llm_config)
        return self._summarization_llm_client

    def _get_extractor(self) -> RubricExtractor:
        """Get or create rubric extractor."""
        if self._extractor is None:
            # Map score_name to corresponding prompt template
            score_name = self.config.score_name
            prompt_map = {
                "token_efficiency": "rubric_extractor_token_efficiency.txt",
                "user_turn_efficiency": "rubric_extractor_user_turn_efficiency.txt",
                "excellence": "rubric_extractor_excellence.txt",
            }
            prompt_filename = prompt_map.get(score_name, "rubric_extractor.txt")

            self._extractor = RubricExtractor(
                llm_client=self._get_extraction_llm_client(),
                prompt_template_path=Path(prompt_filename),
                prompts_dir=self.prompts_dir,
                min_rubrics=self.config.extraction_min_rubrics,
                max_rubrics=self.config.extraction_max_rubrics,
            )
        return self._extractor

    def _get_summarizer(self) -> RubricSummarizer:
        """Get or create rubric summarizer."""
        if self._summarizer is None:
            self._summarizer = RubricSummarizer(
                llm_client=self._get_summarization_llm_client(),
                prompt_template_path=Path("rubric_summarizer.txt"),
                prompts_dir=self.prompts_dir,
                min_rubrics=self.config.min_rubrics,
                max_rubrics=self.config.max_rubrics,
            )
        return self._summarizer

    def _get_semantic_summarizer(self) -> SemanticClusteringSummarizer:
        """Get or create semantic clustering summarizer."""
        if self._semantic_summarizer is None:
            self._semantic_summarizer = SemanticClusteringSummarizer(
                embedding_model=self.config.embedding_model,
                similarity_threshold=self.config.similarity_threshold,
                min_rubrics=self.config.min_rubrics,
                max_rubrics=self.config.max_rubrics,
            )
        return self._semantic_summarizer

    async def train(self) -> RubricList:
        """Execute full training pipeline.

        Returns:
            RubricList containing the final consolidated rubrics.
        """
        logger.info("Starting training pipeline")

        # 1. Load and filter sessions
        logger.info(
            f"Filtering sessions with {self.config.score_name} score >= {self.config.score_threshold}"
        )
        qualified_sessions = self.loader.filter_by_score(
            self.config.score_threshold, self.config.score_name
        )
        total_sessions = self.loader.get_session_count()

        if not qualified_sessions:
            logger.warning("No sessions passed the score threshold")
            return RubricList(rubrics=[])

        # 2. Load chat content for each session
        logger.info("Loading chat sessions")
        # Use sample_size if set, otherwise use max_sessions
        max_sessions = self.config.sample_size if self.config.sample_size is not None else self.config.max_sessions
        if self.config.sample_size is not None:
            logger.info(f"Using sample size limit: {self.config.sample_size} sessions")
        chat_sessions = self.loader.load_sessions(
            qualified_sessions,
            max_sessions=max_sessions,
        )

        if len(chat_sessions) < 3:
            logger.warning(
                f"Only {len(chat_sessions)} sessions loaded. "
                "Consider lowering the score threshold for better results."
            )

        # 3. Extract rubrics from each session (parallel)
        logger.info(f"Extracting rubrics from {len(chat_sessions)} sessions")
        extractor = self._get_extractor()
        rubric_lists = await extractor.extract_batch(
            chat_sessions,
            max_concurrent=self.config.max_concurrent_extractions,
        )

        # Filter out empty results
        non_empty_lists = [r for r in rubric_lists if r]
        if not non_empty_lists:
            logger.warning("No rubrics extracted from any session")
            return RubricList(rubrics=[])

        # 4. Summarize into final rubrics
        logger.info(f"Consolidating rubrics using method: {self.config.summarization_method.value}")
        if self.config.summarization_method == SummarizationMethod.SEMANTIC_CLUSTERING:
            summarizer = self._get_semantic_summarizer()
        else:
            summarizer = self._get_summarizer()
        final_rubrics, consolidation_notes = await summarizer.summarize(non_empty_lists)

        # 5. Create RubricList with metadata
        rubric_list = RubricList(
            version="1.0",
            created_at=datetime.utcnow(),
            training_config=RubricTrainingConfig(
                score_threshold=self.config.score_threshold,
                sessions_used=len(chat_sessions),
                total_sessions=total_sessions,
            ),
            rubrics=final_rubrics,
            consolidation_notes=consolidation_notes,
        )

        logger.info(f"Training complete. Generated {len(final_rubrics)} rubrics")
        return rubric_list

    def save_rubrics(self, rubrics: RubricList, output_path: Path) -> None:
        """Save rubrics to JSON file.

        Args:
            rubrics: RubricList to save.
            output_path: Path for output JSON file.
        """
        rubrics.to_json(output_path)
        logger.info(f"Saved rubrics to {output_path}")
