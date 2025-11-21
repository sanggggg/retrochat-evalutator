"""Training orchestrator for the complete training pipeline."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..models.rubric import Rubric, RubricList, TrainingConfig as RubricTrainingConfig
from ..config import TrainingConfig, LLMConfig
from ..llm.gemini import GeminiClient
from .loader import DatasetLoader
from .extractor import RubricExtractor
from .summarizer import RubricSummarizer

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrate the complete training pipeline."""

    def __init__(
        self,
        dataset_dir: Path,
        manifest_path: Path,
        prompts_dir: Path,
        config: Optional[TrainingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """Initialize trainer.

        Args:
            dataset_dir: Directory containing JSONL session files.
            manifest_path: Path to the dataset manifest JSON file.
            prompts_dir: Directory containing prompt templates.
            config: Training configuration.
            llm_config: LLM configuration.
        """
        self.config = config or TrainingConfig()
        self.llm_config = llm_config or LLMConfig()
        self.prompts_dir = Path(prompts_dir)

        # Initialize components
        self.loader = DatasetLoader(dataset_dir, manifest_path)
        self._llm_client: Optional[GeminiClient] = None
        self._extractor: Optional[RubricExtractor] = None
        self._summarizer: Optional[RubricSummarizer] = None

    def _get_llm_client(self) -> GeminiClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = GeminiClient(self.llm_config)
        return self._llm_client

    def _get_extractor(self) -> RubricExtractor:
        """Get or create rubric extractor."""
        if self._extractor is None:
            self._extractor = RubricExtractor(
                llm_client=self._get_llm_client(),
                prompt_template_path=Path("rubric_extractor.txt"),
                prompts_dir=self.prompts_dir,
            )
        return self._extractor

    def _get_summarizer(self) -> RubricSummarizer:
        """Get or create rubric summarizer."""
        if self._summarizer is None:
            self._summarizer = RubricSummarizer(
                llm_client=self._get_llm_client(),
                prompt_template_path=Path("rubric_summarizer.txt"),
                prompts_dir=self.prompts_dir,
            )
        return self._summarizer

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
        chat_sessions = self.loader.load_sessions(
            qualified_sessions,
            max_sessions=self.config.max_sessions,
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
        logger.info("Consolidating rubrics")
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
