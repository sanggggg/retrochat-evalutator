"""Result aggregator for evaluation scores."""

import logging
from datetime import datetime
from typing import Optional

from ..models.rubric import Rubric
from ..models.evaluation import RubricScore, EvaluationResult, EvaluationSummary

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregate individual rubric scores into final result."""

    def aggregate(
        self,
        session_id: str,
        rubric_scores: list[RubricScore],
        rubrics: list[Rubric],
        rubrics_version: Optional[str] = None,
    ) -> EvaluationResult:
        """Aggregate scores into final evaluation result.

        Args:
            session_id: ID of the evaluated session.
            rubric_scores: List of individual rubric scores.
            rubrics: List of rubrics used for evaluation.
            rubrics_version: Version of rubrics used.

        Returns:
            Complete EvaluationResult with summary.
        """
        # Create rubric weight lookup
        rubric_weights = {r.id: r.weight for r in rubrics}

        # Create evaluation result
        result = EvaluationResult(
            version="1.0",
            session_id=session_id,
            evaluated_at=datetime.utcnow(),
            rubrics_version=rubrics_version,
            rubric_scores=rubric_scores,
        )

        # Calculate summary with weights
        result.calculate_summary(weights=rubric_weights)

        logger.debug(
            f"Aggregated {len(rubric_scores)} scores for session {session_id}. "
            f"Total: {result.summary.total_score}/{result.summary.max_score}"
        )

        return result

    def aggregate_batch(
        self,
        results: list[tuple[str, list[RubricScore]]],
        rubrics: list[Rubric],
        rubrics_version: Optional[str] = None,
    ) -> list[EvaluationResult]:
        """Aggregate scores for multiple sessions.

        Args:
            results: List of (session_id, rubric_scores) tuples.
            rubrics: List of rubrics used for evaluation.
            rubrics_version: Version of rubrics used.

        Returns:
            List of EvaluationResult objects.
        """
        return [
            self.aggregate(session_id, scores, rubrics, rubrics_version)
            for session_id, scores in results
        ]
