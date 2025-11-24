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
    """Statistical metrics for validation.

    Note: When real scores and predicted scores have different scales (e.g., real scores
    have wide range while predicted scores are 1-5), absolute error metrics (MAE, RMSE)
    may not be meaningful. Use scale-independent metrics like correlation and rank correlation instead.
    """

    # Absolute error metrics - may not be meaningful if score scales differ
    mean_absolute_error: float = Field(
        ...,
        description="Mean Absolute Error (MAE). Note: May not be meaningful if score scales differ significantly.",
    )
    root_mean_squared_error: float = Field(
        ...,
        description="Root Mean Squared Error (RMSE). Note: May not be meaningful if score scales differ significantly.",
    )
    mean_error: float = Field(..., description="Mean Error (bias)")
    std_error: Optional[float] = Field(default=None, description="Standard deviation of errors")

    # Scale-independent metrics (recommended when scales differ)
    correlation: Optional[float] = Field(
        default=None, description="Pearson correlation coefficient (scale-independent)"
    )
    rank_correlation: Optional[float] = Field(
        default=None,
        description="Spearman rank correlation coefficient (scale-independent, recommended)",
    )
    r_squared: Optional[float] = Field(
        default=None, description="R-squared (coefficient of determination)"
    )

    # Normalized error metrics (Min-Max normalization applied)
    normalized_mae: Optional[float] = Field(
        default=None,
        description="Normalized MAE after Min-Max scaling to [0,1]. Only meaningful when score scales differ.",
    )
    normalized_rmse: Optional[float] = Field(
        default=None,
        description="Normalized RMSE after Min-Max scaling to [0,1]. Only meaningful when score scales differ.",
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
                    normalized_mae=None,
                    normalized_rmse=None,
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

        # Get score lists for correlation and normalization
        predicted = [r.predicted_score for r in session_results]
        real = [r.real_score for r in session_results]

        # Scale-independent correlation metrics (recommended)
        correlation = cls._calculate_correlation(predicted, real)
        rank_correlation = cls._calculate_spearman_rank_correlation(predicted, real)
        r_squared = correlation**2 if correlation is not None else None

        # Normalized error metrics (Min-Max normalization)
        normalized_mae, normalized_rmse = cls._calculate_normalized_errors(
            predicted, real, abs_errors, sq_errors
        )

        metrics = ValidationMetrics(
            mean_absolute_error=round(mae, 4),
            root_mean_squared_error=round(rmse, 4),
            mean_error=round(mean_err, 4),
            std_error=round(std_err, 4) if std_err is not None else None,
            correlation=round(correlation, 4) if correlation is not None else None,
            rank_correlation=round(rank_correlation, 4) if rank_correlation is not None else None,
            r_squared=round(r_squared, 4) if r_squared is not None else None,
            normalized_mae=round(normalized_mae, 4) if normalized_mae is not None else None,
            normalized_rmse=round(normalized_rmse, 4) if normalized_rmse is not None else None,
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

    @staticmethod
    def _calculate_normalized_errors(
        predicted: list[float],
        real: list[float],
        abs_errors: list[float],
        sq_errors: list[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate normalized MAE and RMSE using Min-Max normalization.

        Both predicted and real scores are normalized to [0, 1] range,
        then errors are calculated on normalized values. This makes metrics
        comparable even when score scales differ significantly.

        Args:
            predicted: List of predicted scores.
            real: List of real scores.
            abs_errors: List of absolute errors (for MAE calculation).
            sq_errors: List of squared errors (for RMSE calculation).

        Returns:
            Tuple of (normalized_mae, normalized_rmse), or (None, None) if cannot calculate.
        """
        if len(predicted) < 2 or len(real) < 2:
            return None, None

        # Check if normalization is needed (i.e., if ranges differ significantly)
        pred_range = max(predicted) - min(predicted)
        real_range = max(real) - min(real)

        # If either range is zero, normalization doesn't make sense
        if pred_range == 0 or real_range == 0:
            return None, None

        # Min-Max normalize both score lists to [0, 1]
        def min_max_normalize(values: list[float]) -> list[float]:
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return [0.0] * len(values)
            return [(v - min_val) / (max_val - min_val) for v in values]

        normalized_pred = min_max_normalize(predicted)
        normalized_real = min_max_normalize(real)

        # Calculate normalized errors
        normalized_abs_errors = [abs(p - r) for p, r in zip(normalized_pred, normalized_real)]
        normalized_sq_errors = [(p - r) ** 2 for p, r in zip(normalized_pred, normalized_real)]

        normalized_mae = mean(normalized_abs_errors)
        normalized_rmse = math.sqrt(mean(normalized_sq_errors))

        return normalized_mae, normalized_rmse

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
