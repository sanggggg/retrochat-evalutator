"""Rubric data models for evaluation criteria."""

from typing import Optional
from datetime import datetime
from pathlib import Path
import json

from pydantic import BaseModel, Field


class Rubric(BaseModel):
    """A single evaluation rubric."""

    id: str = Field(..., description="Unique identifier for the rubric (e.g., rubric_001)")
    name: str = Field(..., description="Short descriptive name (2-5 words)")
    description: str = Field(..., description="What this rubric measures (1-2 sentences)")
    scoring_criteria: str = Field(..., description="How to score from 1 (poor) to 5 (excellent)")
    weight: float = Field(default=1.0, description="Weight for scoring aggregation")
    evidence: Optional[str] = Field(
        default=None, description="Example evidence from training session"
    )

    def format_for_prompt(self) -> str:
        """Format rubric for inclusion in prompts."""
        return f"""Name: {self.name}
Description: {self.description}
Scoring Criteria:
{self.scoring_criteria}"""


class TrainingConfig(BaseModel):
    """Configuration used during rubric training."""

    score_name: str = Field(..., description="Score name used for filtering")
    score_top_percentile: Optional[float] = Field(
        default=None, description="Top percentile cutoff applied to scores"
    )
    sessions_used: int = Field(..., description="Number of sessions used for training")
    total_sessions: int = Field(..., description="Total sessions in dataset")


class RubricList(BaseModel):
    """Container for a list of rubrics with metadata."""

    version: str = Field(default="1.0", description="Schema version")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When rubrics were created"
    )
    training_config: Optional[TrainingConfig] = Field(
        default=None, description="Training configuration if rubrics were learned"
    )
    rubrics: list[Rubric] = Field(default_factory=list, description="List of rubrics")
    consolidation_notes: Optional[str] = Field(
        default=None, description="Notes about how rubrics were consolidated"
    )

    def get_rubric(self, rubric_id: str) -> Optional[Rubric]:
        """Get a rubric by its ID."""
        for rubric in self.rubrics:
            if rubric.id == rubric_id:
                return rubric
        return None

    def to_json(self, path: Path) -> None:
        """Save rubrics to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: Path) -> "RubricList":
        """Load rubrics from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def format_all_for_prompt(self) -> str:
        """Format all rubrics for inclusion in summarizer prompt."""
        formatted = []
        for i, rubric in enumerate(self.rubrics, 1):
            formatted.append(f"--- Rubric {i} ---\n{rubric.format_for_prompt()}")
        return "\n\n".join(formatted)
