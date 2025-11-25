"""Validation result data models for comparing predicted vs real scores."""

from typing import Optional
from datetime import datetime
from pathlib import Path
import json
from statistics import mean, stdev
import math

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
    real_percentile: float = Field(
        ..., description="Percentile rank (0-100) of real score among all real scores"
    )
    error: float = Field(
        ..., description="Difference: predicted_score - real_percentile (note: different scales)"
    )
    absolute_error: float = Field(..., description="Absolute difference")
    squared_error: float = Field(..., description="Squared difference")
    rubric_scores: list[SessionRubricScore] = Field(
        default_factory=list,
        description="Per-rubric scores used to compute the predicted score",
    )


class ValidationMetrics(BaseModel):
    """Statistical metrics for validation.

    Compares predicted_score (raw LLM scores) against real_percentile (0-100).
    Since scales differ, use correlation metrics for meaningful comparison.
    """

    # Error metrics (note: predicted_score and real_percentile have different scales)
    mean_absolute_error: float = Field(
        ...,
        description="Mean Absolute Error (MAE). Note: scales differ between predicted_score and real_percentile.",
    )
    root_mean_squared_error: float = Field(
        ...,
        description="Root Mean Squared Error (RMSE). Note: scales differ.",
    )
    mean_error: float = Field(..., description="Mean Error (bias)")
    std_error: Optional[float] = Field(default=None, description="Standard deviation of errors")

    # Correlation metrics (scale-independent, recommended)
    correlation: Optional[float] = Field(
        default=None,
        description="Pearson correlation between predicted_score and real_percentile",
    )
    rank_correlation: Optional[float] = Field(
        default=None,
        description="Spearman rank correlation (recommended, scale-independent)",
    )
    r_squared: Optional[float] = Field(
        default=None, description="R-squared (coefficient of determination)"
    )

    min_error: float = Field(..., description="Minimum error")
    max_error: float = Field(..., description="Maximum error")


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
                metrics=ValidationMetrics(
                    mean_absolute_error=0,
                    root_mean_squared_error=0,
                    mean_error=0,
                    min_error=0,
                    max_error=0,
                    rank_correlation=None,
                ),
                session_results=[],
            )

        # Calculate metrics from errors (predicted_score vs real_percentile)
        errors = [r.error for r in session_results]
        abs_errors = [r.absolute_error for r in session_results]
        sq_errors = [r.squared_error for r in session_results]

        mae = mean(abs_errors)
        rmse = math.sqrt(mean(sq_errors))
        mean_err = mean(errors)
        std_err = stdev(errors) if len(errors) > 1 else None

        # Get predicted_score and real_percentile for correlation
        predicted_scores = [r.predicted_score for r in session_results]
        real_percentiles = [r.real_percentile for r in session_results]

        # Correlation metrics: predicted_score vs real_percentile
        correlation = cls._calculate_correlation(predicted_scores, real_percentiles)
        rank_correlation = cls._calculate_spearman_rank_correlation(
            predicted_scores, real_percentiles
        )
        r_squared = correlation**2 if correlation is not None else None

        metrics = ValidationMetrics(
            mean_absolute_error=round(mae, 4),
            root_mean_squared_error=round(rmse, 4),
            mean_error=round(mean_err, 4),
            std_error=round(std_err, 4) if std_err is not None else None,
            correlation=round(correlation, 4) if correlation is not None else None,
            rank_correlation=round(rank_correlation, 4) if rank_correlation is not None else None,
            r_squared=round(r_squared, 4) if r_squared is not None else None,
            min_error=round(min(errors), 4),
            max_error=round(max(errors), 4),
        )

        return cls(
            score_name=score_name,
            rubrics_file=rubrics_file,
            total_sessions=len(session_results),
            metrics=metrics,
            session_results=session_results,
        )

    @staticmethod
    def _calculate_correlation(x: list[float], y: list[float]) -> Optional[float]:
        """Calculate Pearson correlation coefficient.

        Args:
            x: First list of values.
            y: Second list of values.

        Returns:
            Correlation coefficient, or None if cannot be calculated.
        """
        if len(x) < 2 or len(x) != len(y):
            return None

        n = len(x)
        mean_x = mean(x)
        mean_y = mean(y)

        # Calculate covariance and standard deviations
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)

        if std_x == 0 or std_y == 0:
            return None

        return covariance / (std_x * std_y)

    @staticmethod
    def _calculate_spearman_rank_correlation(x: list[float], y: list[float]) -> Optional[float]:
        """Calculate Spearman rank correlation coefficient.

        This is scale-independent and measures monotonic relationship between variables.
        Recommended when comparing scores with different scales.

        Args:
            x: First list of values.
            y: Second list of values.

        Returns:
            Spearman rank correlation coefficient, or None if cannot be calculated.
        """
        if len(x) < 2 or len(x) != len(y):
            return None

        # Create rank mappings
        def rank_values(values: list[float]) -> list[float]:
            """Convert values to ranks, handling ties by averaging."""
            sorted_pairs = sorted(enumerate(values), key=lambda p: p[1])
            ranks = [0.0] * len(values)

            i = 0
            while i < len(sorted_pairs):
                # Count how many values are tied at this position
                j = i
                while j < len(sorted_pairs) and sorted_pairs[j][1] == sorted_pairs[i][1]:
                    j += 1

                # Average rank for tied values
                avg_rank = (i + j + 1) / 2

                # Assign rank to all tied positions
                for k in range(i, j):
                    ranks[sorted_pairs[k][0]] = avg_rank

                i = j

            return ranks

        ranks_x = rank_values(x)
        ranks_y = rank_values(y)

        # Calculate Pearson correlation on ranks
        return ValidationReport._calculate_correlation(ranks_x, ranks_y)

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
