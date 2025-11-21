"""Evaluation result data models."""

from typing import Optional
from datetime import datetime
from pathlib import Path
import json
from statistics import mean, median, stdev

from pydantic import BaseModel, Field


class RubricScore(BaseModel):
    """Score for a single rubric."""

    rubric_id: str = Field(..., description="ID of the rubric scored")
    rubric_name: str = Field(..., description="Name of the rubric")
    score: float = Field(..., ge=1, le=5, description="Score from 1-5")
    max_score: float = Field(default=5.0, description="Maximum possible score")
    reasoning: str = Field(..., description="Explanation for the score")

    @property
    def percentage(self) -> float:
        """Get score as percentage."""
        return (self.score / self.max_score) * 100


class EvaluationSummary(BaseModel):
    """Summary statistics for an evaluation."""

    total_score: float = Field(..., description="Weighted total score")
    max_score: float = Field(default=5.0, description="Maximum possible score")
    percentage: float = Field(..., description="Score as percentage")
    rubrics_evaluated: int = Field(..., description="Number of rubrics evaluated")


class EvaluationResult(BaseModel):
    """Complete evaluation result for a session."""

    version: str = Field(default="1.0", description="Schema version")
    session_id: str = Field(..., description="ID of the evaluated session")
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When evaluation was performed"
    )
    rubrics_version: Optional[str] = Field(
        default=None, description="Version of rubrics used"
    )
    rubric_scores: list[RubricScore] = Field(
        default_factory=list, description="Individual rubric scores"
    )
    summary: Optional[EvaluationSummary] = Field(
        default=None, description="Summary statistics"
    )

    def calculate_summary(self, weights: Optional[dict[str, float]] = None) -> EvaluationSummary:
        """Calculate summary statistics from rubric scores."""
        if not self.rubric_scores:
            return EvaluationSummary(
                total_score=0,
                max_score=5.0,
                percentage=0,
                rubrics_evaluated=0,
            )

        if weights:
            total_weight = sum(weights.get(rs.rubric_id, 1.0) for rs in self.rubric_scores)
            weighted_sum = sum(
                rs.score * weights.get(rs.rubric_id, 1.0) for rs in self.rubric_scores
            )
            total_score = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            total_score = mean(rs.score for rs in self.rubric_scores)

        max_score = 5.0
        percentage = (total_score / max_score) * 100

        summary = EvaluationSummary(
            total_score=round(total_score, 2),
            max_score=max_score,
            percentage=round(percentage, 1),
            rubrics_evaluated=len(self.rubric_scores),
        )
        self.summary = summary
        return summary

    def to_json(self, path: Path) -> None:
        """Save evaluation result to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: Path) -> "EvaluationResult":
        """Load evaluation result from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class PerRubricSummary(BaseModel):
    """Summary statistics for a single rubric across multiple sessions."""

    rubric_id: str = Field(..., description="Rubric ID")
    name: str = Field(..., description="Rubric name")
    average: float = Field(..., description="Average score")
    median: float = Field(..., description="Median score")
    std_dev: Optional[float] = Field(default=None, description="Standard deviation")


class BatchEvaluationSummary(BaseModel):
    """Summary for batch evaluation of multiple sessions."""

    total_sessions: int = Field(..., description="Number of sessions evaluated")
    average_score: float = Field(..., description="Average total score")
    median_score: float = Field(..., description="Median total score")
    std_deviation: Optional[float] = Field(default=None, description="Standard deviation")
    score_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of scores (1-5)"
    )
    per_rubric_summary: list[PerRubricSummary] = Field(
        default_factory=list, description="Per-rubric statistics"
    )

    @classmethod
    def from_results(
        cls, results: list[EvaluationResult], rubric_names: Optional[dict[str, str]] = None
    ) -> "BatchEvaluationSummary":
        """Create batch summary from list of evaluation results."""
        if not results:
            return cls(
                total_sessions=0,
                average_score=0,
                median_score=0,
                std_deviation=None,
                score_distribution={},
                per_rubric_summary=[],
            )

        # Collect total scores
        total_scores = []
        for result in results:
            if result.summary:
                total_scores.append(result.summary.total_score)
            elif result.rubric_scores:
                total_scores.append(mean(rs.score for rs in result.rubric_scores))

        # Calculate overall statistics
        avg_score = mean(total_scores) if total_scores else 0
        med_score = median(total_scores) if total_scores else 0
        std_dev = stdev(total_scores) if len(total_scores) > 1 else None

        # Score distribution
        distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        for score in total_scores:
            bucket = str(min(5, max(1, round(score))))
            distribution[bucket] += 1

        # Per-rubric statistics
        rubric_scores_map: dict[str, list[float]] = {}
        for result in results:
            for rs in result.rubric_scores:
                if rs.rubric_id not in rubric_scores_map:
                    rubric_scores_map[rs.rubric_id] = []
                rubric_scores_map[rs.rubric_id].append(rs.score)

        per_rubric = []
        for rubric_id, scores in rubric_scores_map.items():
            name = rubric_names.get(rubric_id, rubric_id) if rubric_names else rubric_id
            per_rubric.append(
                PerRubricSummary(
                    rubric_id=rubric_id,
                    name=name,
                    average=round(mean(scores), 2),
                    median=round(median(scores), 2),
                    std_dev=round(stdev(scores), 2) if len(scores) > 1 else None,
                )
            )

        return cls(
            total_sessions=len(results),
            average_score=round(avg_score, 2),
            median_score=round(med_score, 2),
            std_deviation=round(std_dev, 2) if std_dev else None,
            score_distribution=distribution,
            per_rubric_summary=per_rubric,
        )

    def to_json(self, path: Path) -> None:
        """Save batch summary to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
