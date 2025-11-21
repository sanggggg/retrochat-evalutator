"""Validation result data models for comparing predicted vs real scores."""

from typing import Optional
from datetime import datetime
from pathlib import Path
import json
from statistics import mean, stdev
import math

from pydantic import BaseModel, Field


class SessionValidationResult(BaseModel):
    """Validation result for a single session."""

    session_id: str = Field(..., description="ID of the session")
    file: str = Field(..., description="Filename of the session")
    predicted_score: float = Field(..., description="Score predicted by LLM evaluation")
    real_score: float = Field(..., description="Actual score from manifest")
    error: float = Field(..., description="Difference: predicted - real")
    absolute_error: float = Field(..., description="Absolute difference")
    squared_error: float = Field(..., description="Squared difference")


class ValidationMetrics(BaseModel):
    """Statistical metrics for validation."""

    mean_absolute_error: float = Field(..., description="Mean Absolute Error (MAE)")
    root_mean_squared_error: float = Field(..., description="Root Mean Squared Error (RMSE)")
    mean_error: float = Field(..., description="Mean Error (bias)")
    std_error: Optional[float] = Field(default=None, description="Standard deviation of errors")
    correlation: Optional[float] = Field(
        default=None, description="Pearson correlation coefficient"
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
                ),
                session_results=[],
            )

        # Calculate metrics
        errors = [r.error for r in session_results]
        abs_errors = [r.absolute_error for r in session_results]
        sq_errors = [r.squared_error for r in session_results]

        mae = mean(abs_errors)
        rmse = math.sqrt(mean(sq_errors))
        mean_err = mean(errors)
        std_err = stdev(errors) if len(errors) > 1 else None

        # Calculate correlation and R-squared
        predicted = [r.predicted_score for r in session_results]
        real = [r.real_score for r in session_results]
        correlation = cls._calculate_correlation(predicted, real)
        r_squared = correlation**2 if correlation is not None else None

        metrics = ValidationMetrics(
            mean_absolute_error=round(mae, 4),
            root_mean_squared_error=round(rmse, 4),
            mean_error=round(mean_err, 4),
            std_error=round(std_err, 4) if std_err is not None else None,
            correlation=round(correlation, 4) if correlation is not None else None,
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
