"""Training orchestrator for the complete training pipeline."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..models.rubric import Rubric, RubricList, TrainingConfig as RubricTrainingConfig
from ..models.chat_session import ChatSession
from ..config import (
    TrainingConfig,
    ExtractionLLMConfig,
    SummarizationLLMConfig,
    SummarizationMethod,
    Config,
    EvaluationConfig,
    EvaluationLLMConfig,
    RateLimiterConfig,
)
from ..llm.gemini import GeminiClient
from .loader import DatasetLoader
from .extractor import RubricExtractor
from .llm_summarizer import RubricSummarizer
from .semantic_summarizer import SemanticClusteringSummarizer
from .visualizer import save_clustering_visualization

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
        full_config: Optional[Config] = None,
        evaluation_config: Optional[EvaluationConfig] = None,
        evaluation_llm_config: Optional[EvaluationLLMConfig] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
    ):
        """Initialize trainer.

        Args:
            dataset_dir: Directory containing JSONL session files.
            manifest_path: Path to the dataset manifest JSON file.
            prompts_dir: Directory containing prompt templates.
            config: Training configuration.
            extraction_llm_config: LLM configuration for rubric extraction.
            summarization_llm_config: LLM configuration for rubric summarization.
            full_config: Full configuration object for metadata (optional).
            evaluation_config: Evaluation configuration for validation (optional).
            evaluation_llm_config: LLM configuration for evaluation (optional).
            rate_limiter_config: Shared rate limiter configuration for all LLM clients.
        """
        self.config = config or TrainingConfig()
        self.extraction_llm_config = extraction_llm_config or ExtractionLLMConfig()
        self.summarization_llm_config = summarization_llm_config or SummarizationLLMConfig()
        self.evaluation_config = evaluation_config
        self.evaluation_llm_config = evaluation_llm_config
        self.rate_limiter_config = rate_limiter_config
        self.full_config = full_config
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
            self._extraction_llm_client = GeminiClient(
                self.extraction_llm_config,
                rate_limiter_config=self.rate_limiter_config,
            )
        return self._extraction_llm_client

    def _get_summarization_llm_client(self) -> GeminiClient:
        """Get or create LLM client for summarization."""
        if self._summarization_llm_client is None:
            self._summarization_llm_client = GeminiClient(
                self.summarization_llm_config,
                rate_limiter_config=self.rate_limiter_config,
            )
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
                umap_n_neighbors=self.config.umap_n_neighbors,
                umap_n_components=self.config.umap_n_components,
                umap_metric=self.config.umap_metric,
                min_cluster_size=self.config.min_cluster_size,
                min_rubrics=self.config.min_rubrics,
                max_rubrics=self.config.max_rubrics,
            )
        return self._semantic_summarizer

    async def train(self) -> tuple[RubricList, dict[str, list[Rubric]], list[ChatSession]]:
        """Execute full training pipeline.

        Returns:
            Tuple of (RubricList containing the final consolidated rubrics,
                     dict mapping session_id to list of extracted rubrics,
                     list of ChatSession objects used for training).
        """
        logger.info("Starting training pipeline")

        # 1. Load and filter sessions (only training split if available)
        qualified_sessions = []
        if self.config.score_top_percentile is not None:
            logger.info(
                f"Filtering training sessions to top {self.config.score_top_percentile}% "
                f"by {self.config.score_name} score"
            )
            qualified_sessions = self.loader.filter_by_percentile(
                self.config.score_top_percentile, self.config.score_name, split="training"
            )
            if not qualified_sessions:
                logger.warning("No training split found, using all sessions (legacy dataset?)")
                qualified_sessions = self.loader.filter_by_percentile(
                    self.config.score_top_percentile, self.config.score_name, split=None
                )
        else:
            logger.info("No score filter configured, using all available sessions")
            qualified_sessions = self.loader.filter_by_split("training")
            if not qualified_sessions:
                qualified_sessions = self.loader.load_manifest().sessions
        total_sessions = self.loader.get_session_count()

        if not qualified_sessions:
            logger.warning("No sessions satisfied the configured score filter")
            return RubricList(rubrics=[]), {}, []

        # 2. Load chat content for each session
        logger.info("Loading chat sessions")
        # Note: max_sessions and sample_size are no longer in config.
        # They should be specified when generating dataset.json via generate_manifest.py
        # For training, all sessions in the manifest are used (no additional limiting).
        chat_sessions = self.loader.load_sessions(
            qualified_sessions,
            max_sessions=None,  # Use all sessions from manifest
        )

        if len(chat_sessions) < 3:
            if self.config.score_top_percentile is not None:
                guidance = (
                    f"Consider increasing the percentile cutoff "
                    f"(currently top {self.config.score_top_percentile}%) for better coverage."
                )
            else:
                guidance = "Consider adding more sessions to the dataset."
            logger.warning(f"Only {len(chat_sessions)} sessions loaded. {guidance}")

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
            return RubricList(rubrics=[]), {}, chat_sessions

        # Create mapping of session_id to rubrics
        raw_rubrics_map: dict[str, list[Rubric]] = {}
        for session, rubric_list in zip(chat_sessions, rubric_lists):
            if rubric_list:  # Only include non-empty lists
                raw_rubrics_map[session.session_id] = rubric_list

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
                score_name=self.config.score_name,
                score_top_percentile=self.config.score_top_percentile,
                sessions_used=len(chat_sessions),
                total_sessions=total_sessions,
            ),
            rubrics=final_rubrics,
            consolidation_notes=consolidation_notes,
        )

        logger.info(f"Training complete. Generated {len(final_rubrics)} rubrics")
        return rubric_list, raw_rubrics_map, chat_sessions

    async def save_rubrics(
        self,
        rubrics: RubricList,
        output_path: Path,
        raw_rubrics_map: Optional[dict[str, list[Rubric]]] = None,
        validate: bool = True,
    ) -> Path:
        """Save rubrics to organized folder structure.

        Creates a folder structure:
        - output_path/train-result-YYYY-MM-DD-HHMMSS/
          - rubrics.json (final consolidated rubrics)
          - metadata.json (training config and metadata)
          - raw-rubrics.json (session-to-rubrics mapping)

        Args:
            rubrics: RubricList to save.
            output_path: Base output directory (will create subfolder).
            raw_rubrics_map: Optional mapping of session_id to extracted rubrics.
            validate: Whether to perform validation after saving (default: True).

        Returns:
            Path to the created training result folder.
        """
        # Check if output_path exists and is a file (not a directory)
        if output_path.exists() and output_path.is_file():
            raise ValueError(
                f"Output path '{output_path}' exists as a file. "
                "Please remove it or specify a different directory path."
            )

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        result_folder = output_path / f"train-result-{timestamp}"
        result_folder.mkdir(parents=True, exist_ok=True)

        # 1. Save rubrics.json
        rubrics_path = result_folder / "rubrics.json"
        rubrics.to_json(rubrics_path)
        logger.info(f"Saved rubrics to {rubrics_path}")

        # 2. Save metadata.json
        metadata = {
            "created_at": rubrics.created_at.isoformat(),
            "version": rubrics.version,
            "training_config": {
                "score_name": self.config.score_name,
                "score_top_percentile": self.config.score_top_percentile,
                "sessions_used": (
                    rubrics.training_config.sessions_used if rubrics.training_config else 0
                ),
                "total_sessions": (
                    rubrics.training_config.total_sessions if rubrics.training_config else 0
                ),
                "summarization_method": self.config.summarization_method.value,
                "min_rubrics": self.config.min_rubrics,
                "max_rubrics": self.config.max_rubrics,
                "extraction_min_rubrics": self.config.extraction_min_rubrics,
                "extraction_max_rubrics": self.config.extraction_max_rubrics,
                "max_concurrent_extractions": self.config.max_concurrent_extractions,
            },
            "llm_config": {
                "extraction": {
                    "model_name": self.extraction_llm_config.model_name,
                    "temperature": self.extraction_llm_config.temperature,
                    "max_tokens": self.extraction_llm_config.max_tokens,
                },
                "summarization": {
                    "model_name": self.summarization_llm_config.model_name,
                    "temperature": self.summarization_llm_config.temperature,
                    "max_tokens": self.summarization_llm_config.max_tokens,
                },
            },
            "consolidation_notes": rubrics.consolidation_notes,
            "evaluations": [],  # Will be populated later when evaluations are run
        }

        # Add semantic clustering config if used
        if self.config.summarization_method == SummarizationMethod.SEMANTIC_CLUSTERING:
            metadata["training_config"]["embedding_model"] = self.config.embedding_model
            metadata["training_config"]["similarity_threshold"] = self.config.similarity_threshold

        # Add full config paths if available
        if self.full_config:
            if self.full_config.dataset_dir:
                metadata["dataset_dir"] = str(self.full_config.dataset_dir)
            if self.full_config.dataset_manifest:
                metadata["dataset_manifest"] = str(self.full_config.dataset_manifest)

        metadata_path = result_folder / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")

        # 3. Save raw-rubrics.json
        if raw_rubrics_map:
            raw_rubrics_data = {}
            for session_id, rubric_list in raw_rubrics_map.items():
                raw_rubrics_data[session_id] = [
                    rubric.model_dump(mode="json") for rubric in rubric_list
                ]

            raw_rubrics_path = result_folder / "raw-rubrics.json"
            with open(raw_rubrics_path, "w", encoding="utf-8") as f:
                json.dump(raw_rubrics_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved raw rubrics to {raw_rubrics_path}")
        else:
            # Create empty file if no raw rubrics provided
            raw_rubrics_path = result_folder / "raw-rubrics.json"
            with open(raw_rubrics_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            logger.info(f"Created empty raw-rubrics.json at {raw_rubrics_path}")

        # 4. Generate clustering visualization if semantic clustering was used
        if (
            self.config.summarization_method == SummarizationMethod.SEMANTIC_CLUSTERING
            and self._semantic_summarizer is not None
            and self._semantic_summarizer.last_embeddings is not None
            and self._semantic_summarizer.last_clusters is not None
            and self._semantic_summarizer.last_all_rubrics is not None
        ):
            try:
                save_clustering_visualization(
                    result_folder,
                    self._semantic_summarizer.last_embeddings,
                    self._semantic_summarizer.last_clusters,
                    self._semantic_summarizer.last_all_rubrics,
                    self._semantic_summarizer.last_cluster_info,
                    umap_embeddings=self._semantic_summarizer.last_umap_embeddings,
                )
            except Exception as e:
                logger.warning(f"Failed to generate clustering visualization: {e}")

        # 5. Perform validation if requested
        if validate:
            try:
                logger.info("Starting validation of trained rubrics...")

                # Get dataset_dir and manifest_path from loader or full_config
                dataset_dir = self.loader.dataset_dir
                manifest_path = self.loader.manifest_path

                # Use evaluation config from full_config if available, otherwise use defaults
                eval_config = self.evaluation_config
                eval_llm_config = self.evaluation_llm_config

                if self.full_config:
                    if eval_config is None:
                        eval_config = self.full_config.evaluation
                    if eval_llm_config is None:
                        eval_llm_config = self.full_config.evaluation_llm

                # Lazy import to avoid circular dependency
                from ..validation.validator import Validator

                # Initialize validator
                validator = Validator(
                    dataset_dir=dataset_dir,
                    manifest_path=manifest_path,
                    rubrics_path=rubrics_path,
                    prompts_dir=self.prompts_dir,
                    score_name=self.config.score_name,
                    config=eval_config,
                    llm_config=eval_llm_config,
                    rate_limiter_config=self.rate_limiter_config,
                )

                # Run validation
                validation_report = await validator.validate()

                # Persist per-session validation results for downstream analysis
                session_results_path = result_folder / "session-validation-results.json"
                session_results_data = [
                    result.model_dump(mode="json") for result in validation_report.session_results
                ]
                with open(session_results_path, "w", encoding="utf-8") as f:
                    json.dump(session_results_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved session validation results to {session_results_path}")

                # Convert validation report to dict for metadata
                validation_metrics = validation_report.metrics
                validation_data = {
                    "created_at": validation_report.created_at.isoformat(),
                    "score_name": validation_report.score_name,
                    "total_sessions": validation_report.total_sessions,
                    "session_results_file": session_results_path.name,
                    "session_results_count": len(session_results_data),
                    "metrics": {
                        "mean_absolute_error": validation_metrics.mean_absolute_error,
                        "root_mean_squared_error": validation_metrics.root_mean_squared_error,
                        "mean_error": validation_metrics.mean_error,
                        "std_error": validation_metrics.std_error,
                        "correlation": validation_metrics.correlation,
                        "rank_correlation": validation_metrics.rank_correlation,
                        "r_squared": validation_metrics.r_squared,
                        "min_error": validation_metrics.min_error,
                        "max_error": validation_metrics.max_error,
                    },
                }

                # Update metadata with validation results
                metadata["validation"] = validation_data

                # Save updated metadata.json
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated metadata with validation results at {metadata_path}")

                # Log validation summary
                metrics = validation_metrics

                def _fmt_optional(value: float | None) -> str:
                    return f"{value:.4f}" if value is not None else "N/A"

                logger.info(
                    "Validation complete: "
                    f"Correlation={_fmt_optional(metrics.correlation)}, "
                    f"Rank Correlation={_fmt_optional(metrics.rank_correlation)}, "
                    f"R²={_fmt_optional(metrics.r_squared)}, "
                    f"MAE={metrics.mean_absolute_error:.4f}"
                )

            except Exception as e:
                logger.warning(f"Validation failed: {e}", exc_info=True)
                # Add error info to metadata
                metadata["validation"] = {
                    "error": str(e),
                    "status": "failed",
                }
                # Save updated metadata with error
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

        return result_folder
