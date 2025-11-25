"""Validator for comparing LLM-predicted scores against real scores."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..models.rubric import Rubric, RubricList
from ..models.chat_session import ChatSession
from ..models.evaluation import EvaluationResult
from ..models.validation import SessionRubricScore, SessionValidationResult, ValidationReport
from ..config import EvaluationConfig, EvaluationLLMConfig, RateLimiterConfig
from ..training.loader import DatasetLoader, SessionInfo
from ..evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


@dataclass
class _IntermediateResult:
    """Intermediate result before percentile calculation."""

    session_id: str
    file: str
    predicted_score: float
    real_score: float
    rubric_scores: list[SessionRubricScore]


class Validator:
    """Validate evaluation accuracy by comparing predicted vs real scores."""

    def __init__(
        self,
        dataset_dir: Path,
        manifest_path: Path,
        rubrics_path: Path,
        prompts_dir: Path,
        score_name: str = "default",
        config: Optional[EvaluationConfig] = None,
        llm_config: Optional[EvaluationLLMConfig] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
    ):
        """Initialize validator.

        Args:
            dataset_dir: Directory containing JSONL session files.
            manifest_path: Path to the dataset manifest JSON file.
            rubrics_path: Path to the rubrics JSON file.
            prompts_dir: Directory containing prompt templates.
            score_name: Name of the score to use for comparison.
            config: Evaluation configuration.
            llm_config: LLM configuration for evaluation.
            rate_limiter_config: Shared rate limiter configuration for all LLM clients.
        """
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = Path(manifest_path)
        self.rubrics_path = Path(rubrics_path)
        self.prompts_dir = Path(prompts_dir)
        self.score_name = score_name

        self.config = config or EvaluationConfig()
        self.llm_config = llm_config or EvaluationLLMConfig()
        self.rate_limiter_config = rate_limiter_config

        # Initialize components
        self._loader: Optional[DatasetLoader] = None
        self._evaluator: Optional[Evaluator] = None
        self._rubric_list: Optional[RubricList] = None

    def _get_loader(self) -> DatasetLoader:
        """Get or create dataset loader."""
        if self._loader is None:
            self._loader = DatasetLoader(self.dataset_dir, self.manifest_path)
        return self._loader

    def _get_evaluator(self) -> Evaluator:
        """Get or create evaluator."""
        if self._evaluator is None:
            self._evaluator = Evaluator(
                prompts_dir=self.prompts_dir,
                config=self.config,
                llm_config=self.llm_config,
                rate_limiter_config=self.rate_limiter_config,
            )
        return self._evaluator

    def _get_rubrics(self) -> RubricList:
        """Get or create rubric list."""
        if self._rubric_list is None:
            self._rubric_list = RubricList.from_json(self.rubrics_path)
        return self._rubric_list

    async def validate(self) -> ValidationReport:
        """Execute full validation pipeline.

        Returns:
            ValidationReport comparing predicted vs real scores.
        """
        # Load components
        loader = self._get_loader()
        evaluator = self._get_evaluator()
        rubric_list = self._get_rubrics()

        # Get validation sessions from manifest
        manifest = loader.load_manifest()
        session_infos = manifest.sessions

        # Filter to validation split only (if available)
        validation_sessions = [s for s in session_infos if s.split == "validation"]
        if validation_sessions:
            session_infos = validation_sessions
            logger.info(f"Using validation split: {len(session_infos)} sessions")
        else:
            logger.warning("No validation split found, using all sessions (legacy dataset?)")
            # Fall back to all sessions if no split is available

        # Filter to sessions that have the specified score
        session_infos = [s for s in session_infos if s.get_score(self.score_name) is not None]

        if not session_infos:
            logger.warning(f"No sessions found with score '{self.score_name}'")
            return ValidationReport.from_results(
                session_results=[],
                score_name=self.score_name,
                rubrics_file=str(self.rubrics_path),
            )

        logger.info(
            f"Validating {len(session_infos)} sessions against {len(rubric_list.rubrics)} rubrics"
        )

        # Phase 1: Evaluate all sessions and collect intermediate results
        intermediate_results: list[_IntermediateResult] = []

        for i, session_info in enumerate(session_infos, 1):
            logger.info(f"Processing session {i}/{len(session_infos)}: {session_info.file}")

            try:
                result = await self._evaluate_session(
                    session_info=session_info,
                    rubrics=rubric_list.rubrics,
                    rubrics_version=rubric_list.version,
                    evaluator=evaluator,
                    loader=loader,
                )
                intermediate_results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate session {session_info.file}: {e}")

        # Phase 2: Calculate percentiles and create final results
        session_results = self._calculate_percentiles_and_create_results(intermediate_results)

        # Generate report
        report = ValidationReport.from_results(
            session_results=session_results,
            score_name=self.score_name,
            rubrics_file=str(self.rubrics_path),
        )

        # Build metrics summary (comparing predicted_score vs real_percentile)
        metrics = report.metrics
        metric_parts = [
            (
                f"Pearson Correlation: {metrics.correlation:.4f}"
                if metrics.correlation is not None
                else None
            ),
            (
                f"Spearman Rank Correlation: {metrics.rank_correlation:.4f}"
                if metrics.rank_correlation is not None
                else None
            ),
            f"R²: {metrics.r_squared:.4f}" if metrics.r_squared is not None else None,
        ]

        logger.info(
            f"Validation complete. {' | '.join([m for m in metric_parts if m is not None])}"
        )

        return report

    async def _evaluate_session(
        self,
        session_info: SessionInfo,
        rubrics: list[Rubric],
        rubrics_version: Optional[str],
        evaluator: Evaluator,
        loader: DatasetLoader,
    ) -> _IntermediateResult:
        """Evaluate a single session and return intermediate result.

        Args:
            session_info: Session information from manifest.
            rubrics: List of rubrics to evaluate against.
            rubrics_version: Version of rubrics.
            evaluator: Evaluator instance.
            loader: Dataset loader instance.

        Returns:
            _IntermediateResult with scores (before percentile calculation).
        """
        # Load session
        session = loader.load_session(session_info)

        # Evaluate session
        eval_result = await evaluator.evaluate(
            rubrics=rubrics,
            session=session,
            rubrics_version=rubrics_version,
        )

        # Get scores
        real_score = session_info.get_score(self.score_name)
        predicted_score = eval_result.summary.total_score

        per_rubric_scores = [
            SessionRubricScore(
                rubric_id=rs.rubric_id,
                rubric_name=rs.rubric_name,
                score=rs.score,
                max_score=rs.max_score,
            )
            for rs in eval_result.rubric_scores
        ]

        return _IntermediateResult(
            session_id=session.session_id,
            file=session_info.file,
            predicted_score=predicted_score,
            real_score=real_score,
            rubric_scores=per_rubric_scores,
        )

    def _calculate_percentiles_and_create_results(
        self,
        intermediate_results: list[_IntermediateResult],
    ) -> list[SessionValidationResult]:
        """Calculate real_percentile and create final SessionValidationResults.

        Args:
            intermediate_results: List of intermediate results with raw scores.

        Returns:
            List of SessionValidationResult comparing predicted_score vs real_percentile.
        """
        if not intermediate_results:
            return []

        # Extract real scores and calculate their percentile ranks
        real_scores = [r.real_score for r in intermediate_results]
        real_percentiles = self._calculate_percentile_ranks(real_scores)

        # Create final results comparing predicted_score vs real_percentile
        session_results: list[SessionValidationResult] = []
        for i, ir in enumerate(intermediate_results):
            real_pct = real_percentiles[i]

            # Calculate error metrics: predicted_score vs real_percentile
            # Note: these have different scales, so absolute error metrics may not be meaningful
            error = ir.predicted_score - real_pct
            absolute_error = abs(error)
            squared_error = error**2

            session_results.append(
                SessionValidationResult(
                    session_id=ir.session_id,
                    file=ir.file,
                    predicted_score=round(ir.predicted_score, 4),
                    real_score=ir.real_score,
                    real_percentile=round(real_pct, 4),
                    error=round(error, 4),
                    absolute_error=round(absolute_error, 4),
                    squared_error=round(squared_error, 4),
                    rubric_scores=ir.rubric_scores,
                )
            )

        return session_results

    @staticmethod
    def _calculate_percentile_ranks(scores: list[float]) -> list[float]:
        """Calculate percentile ranks for a list of scores.

        Uses the 'rank' method where percentile = (rank / n) * 100.
        Higher scores get higher percentiles.

        Args:
            scores: List of scores.

        Returns:
            List of percentile ranks (0-100) corresponding to each score.
        """
        n = len(scores)
        if n == 0:
            return []
        if n == 1:
            return [50.0]  # Single item gets median percentile

        # Create sorted index pairs (original_index, score)
        indexed_scores = list(enumerate(scores))
        # Sort by score ascending
        sorted_by_score = sorted(indexed_scores, key=lambda x: x[1])

        # Assign ranks with tie handling (average rank for ties)
        percentiles = [0.0] * n
        i = 0
        while i < n:
            # Find all items with the same score (ties)
            j = i
            while j < n and sorted_by_score[j][1] == sorted_by_score[i][1]:
                j += 1

            # Average rank for tied items (1-based ranks)
            avg_rank = (i + j + 1) / 2  # Average of ranks from i+1 to j

            # Convert rank to percentile: (rank / n) * 100
            percentile = (avg_rank / n) * 100

            # Assign percentile to all tied items
            for k in range(i, j):
                original_index = sorted_by_score[k][0]
                percentiles[original_index] = percentile

            i = j

        return percentiles

    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to JSON file.

        Args:
            report: ValidationReport to save.
            output_path: Path for output JSON file.
        """
        report.to_json(output_path)
        logger.info(f"Saved validation report to {output_path}")
