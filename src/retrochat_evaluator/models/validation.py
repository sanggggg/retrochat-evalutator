"""Validation result data models for comparing predicted vs real scores."""

from typing import Optional
from datetime import datetime
from pathlib import Path
import json

from scipy import stats
from pydantic import BaseModel, Field


class SessionRubricScore(BaseModel):
    """Score information for a single rubric within session validation results."""

    rubric_id: str = Field(..., description="ID of the rubric")
    rubric_name: str = Field(..., description="Human-readable rubric name")
    score: float = Field(..., description="Score predicted for this rubric")
    max_score: float = Field(..., description="Maximum possible score for this rubric")


class SessionValidationResult(BaseModel):
    """Validation result for a single session."""

    session_id: str = Field(..., description="ID of the session")
    file: str = Field(..., description="Filename of the session")
    predicted_score: float = Field(..., description="Score predicted by LLM evaluation")
    real_score: float = Field(..., description="Actual score from manifest")
    rubric_scores: list[SessionRubricScore] = Field(
        default_factory=list,
        description="Per-rubric scores used to compute the predicted score",
    )


class ValidationMetrics(BaseModel):
    """Statistical metrics for validation."""

    rank_correlation: Optional[float] = Field(
        default=None,
        description="Kendall rank correlation (tau) between predicted_score and real_score",
    )
    p_value: Optional[float] = Field(
        default=None,
        description="P-value for Kendall rank correlation (statistical significance)",
    )


class ValidationReport(BaseModel):
    """Complete validation report comparing predicted vs real scores."""

    version: str = Field(default="1.0", description="Schema version")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When validation was performed"
    )
    score_name: str = Field(..., description="Name of the score type used for comparison")
    rubrics_file: str = Field(..., description="Path to rubrics file used")
    total_sessions: int = Field(..., description="Number of sessions validated")
    metrics: ValidationMetrics = Field(..., description="Validation metrics")
    session_results: list[SessionValidationResult] = Field(
        default_factory=list, description="Per-session validation results"
    )

    @classmethod
    def from_results(
        cls,
        session_results: list[SessionValidationResult],
        score_name: str,
        rubrics_file: str,
    ) -> "ValidationReport":
        """Create validation report from session results.

        Args:
            session_results: List of per-session validation results.
            score_name: Name of the score type used.
            rubrics_file: Path to rubrics file used.

        Returns:
            ValidationReport with calculated metrics.
        """
        if not session_results:
            return cls(
                score_name=score_name,
                rubrics_file=rubrics_file,
                total_sessions=0,
                metrics=ValidationMetrics(rank_correlation=None, p_value=None),
                session_results=[],
            )

        # Get predicted_score and real_score for correlation
        predicted_scores = [r.predicted_score for r in session_results]
        real_scores = [r.real_score for r in session_results]

        # Calculate Kendall rank correlation with p-value
        rank_correlation, p_value = cls._calculate_kendall_rank_correlation(
            predicted_scores, real_scores
        )

        metrics = ValidationMetrics(
            rank_correlation=round(rank_correlation, 4) if rank_correlation is not None else None,
            p_value=round(p_value, 6) if p_value is not None else None,
        )

        return cls(
            score_name=score_name,
            rubrics_file=rubrics_file,
            total_sessions=len(session_results),
            metrics=metrics,
            session_results=session_results,
        )

    @staticmethod
    def _calculate_kendall_rank_correlation(
        x: list[float], y: list[float]
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate Kendall Tau rank correlation coefficient using scipy.

        This is scale-independent and measures monotonic relationship between variables.
        Recommended when comparing scores with different scales.

        Args:
            x: First list of values.
            y: Second list of values.

        Returns:
            Tuple of (correlation coefficient, p-value), or (None, None) if cannot be calculated.
        """
        if len(x) < 2 or len(x) != len(y):
            return None, None

        try:
            # Use scipy.stats.kendalltau for robust calculation
            result = stats.kendalltau(x, y)

            # Handle NaN values (e.g., when one variable has zero variance)
            import math

            if math.isnan(result.statistic) or math.isnan(result.pvalue):
                return None, None

            return result.statistic, result.pvalue
        except Exception:
            return None, None

    def to_json(self, path: Path) -> None:
        """Save validation report to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: Path) -> "ValidationReport":
        """Load validation report from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
