"""Evaluation orchestrator for the complete evaluation pipeline."""

import logging
from pathlib import Path
from typing import Optional

from ..models.rubric import Rubric, RubricList
from ..models.chat_session import ChatSession
from ..models.evaluation import EvaluationResult, BatchEvaluationSummary
from ..config import EvaluationConfig, EvaluationLLMConfig, RateLimiterConfig
from ..llm.gemini import GeminiClient
from .judge import JudgePromptGenerator
from .scorer import RubricScorer
from .aggregator import ResultAggregator

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrate the complete evaluation pipeline."""

    def __init__(
        self,
        prompts_dir: Path,
        config: Optional[EvaluationConfig] = None,
        llm_config: Optional[EvaluationLLMConfig] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
    ):
        """Initialize evaluator.

        Args:
            prompts_dir: Directory containing prompt templates.
            config: Evaluation configuration.
            llm_config: LLM configuration for evaluation.
            rate_limiter_config: Shared rate limiter configuration for all LLM clients.
        """
        self.config = config or EvaluationConfig()
        self.llm_config = llm_config or EvaluationLLMConfig()
        self.rate_limiter_config = rate_limiter_config
        self.prompts_dir = Path(prompts_dir)

        # Initialize components
        self._llm_client: Optional[GeminiClient] = None
        self._prompt_generator: Optional[JudgePromptGenerator] = None
        self._scorer: Optional[RubricScorer] = None
        self._aggregator = ResultAggregator()

    def _get_llm_client(self) -> GeminiClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = GeminiClient(
                self.llm_config,
                rate_limiter_config=self.rate_limiter_config,
            )
        return self._llm_client

    def _get_prompt_generator(self) -> JudgePromptGenerator:
        """Get or create prompt generator."""
        if self._prompt_generator is None:
            self._prompt_generator = JudgePromptGenerator(
                template_path=Path("judge_template.txt"),
                prompts_dir=self.prompts_dir,
            )
        return self._prompt_generator

    def _get_scorer(self) -> RubricScorer:
        """Get or create scorer."""
        if self._scorer is None:
            self._scorer = RubricScorer(
                llm_client=self._get_llm_client(),
                config=self.config,
            )
        return self._scorer

    async def evaluate(
        self,
        rubrics: list[Rubric],
        session: ChatSession,
        rubrics_version: Optional[str] = None,
    ) -> EvaluationResult:
        """Execute full evaluation pipeline for a single session.

        Args:
            rubrics: List of rubrics to evaluate against.
            session: Chat session to evaluate.
            rubrics_version: Version of rubrics being used.

        Returns:
            EvaluationResult with scores for all rubrics.
        """
        logger.info(f"Evaluating session {session.session_id} against {len(rubrics)} rubrics")

        # 1. Generate judge prompts for each rubric
        prompt_generator = self._get_prompt_generator()
        rubric_prompts = prompt_generator.generate_all(rubrics, session)

        # 2. Score against all rubrics (parallel LLM calls)
        scorer = self._get_scorer()
        rubric_scores = await scorer.score_all(rubric_prompts)

        # 3. Aggregate into final result
        result = self._aggregator.aggregate(
            session_id=session.session_id,
            rubric_scores=rubric_scores,
            rubrics=rubrics,
            rubrics_version=rubrics_version,
        )

        logger.info(
            f"Evaluation complete for {session.session_id}. "
            f"Score: {result.summary.total_score}/{result.summary.max_score} "
            f"({result.summary.percentage}%)"
        )

        return result

    async def evaluate_batch(
        self,
        rubrics: list[Rubric],
        sessions: list[ChatSession],
        rubrics_version: Optional[str] = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple sessions.

        Args:
            rubrics: List of rubrics to evaluate against.
            sessions: List of chat sessions to evaluate.
            rubrics_version: Version of rubrics being used.

        Returns:
            List of EvaluationResult objects.
        """
        logger.info(f"Batch evaluating {len(sessions)} sessions")

        results = []
        for i, session in enumerate(sessions, 1):
            logger.info(f"Processing session {i}/{len(sessions)}: {session.session_id}")
            result = await self.evaluate(rubrics, session, rubrics_version)
            results.append(result)

        return results

    def save_result(self, result: EvaluationResult, output_path: Path) -> None:
        """Save evaluation result to JSON file.

        Args:
            result: EvaluationResult to save.
            output_path: Path for output JSON file.
        """
        result.to_json(output_path)
        logger.info(f"Saved evaluation result to {output_path}")

    def save_batch_results(
        self,
        results: list[EvaluationResult],
        output_dir: Path,
        rubric_names: Optional[dict[str, str]] = None,
    ) -> Path:
        """Save batch evaluation results and summary.

        Args:
            results: List of evaluation results.
            output_dir: Directory for output files.
            rubric_names: Optional mapping of rubric IDs to names.

        Returns:
            Path to the summary file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual results
        for result in results:
            result_path = output_dir / f"{result.session_id}_result.json"
            result.to_json(result_path)

        # Generate and save summary
        summary = BatchEvaluationSummary.from_results(results, rubric_names)
        summary_path = output_dir / "summary.json"
        summary.to_json(summary_path)

        logger.info(f"Saved {len(results)} results and summary to {output_dir}")
        return summary_path
