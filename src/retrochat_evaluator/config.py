"""Configuration management for Retrochat Evaluator."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv


class SummarizationMethod(str, Enum):
    """Method for consolidating extracted rubrics."""

    LLM = "llm"  # Use LLM prompt-based summarization
    SEMANTIC_CLUSTERING = "semantic_clustering"  # Use embedding + HAC clustering


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-pro-preview-06-05"
    temperature: float = 0.3
    max_tokens: int = 4096
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if self.api_key is None:
            load_dotenv()
            self.api_key = os.getenv("GOOGLE_API_KEY")


@dataclass
class TrainingConfig:
    """Configuration for training module."""

    score_threshold: float = 4.0
    score_name: str = "default"
    max_sessions: Optional[int] = None
    max_concurrent_extractions: int = 5
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # Summarization method configuration
    summarization_method: SummarizationMethod = SummarizationMethod.LLM

    # Semantic clustering configuration (used when summarization_method == SEMANTIC_CLUSTERING)
    embedding_model: str = "models/text-embedding-004"
    similarity_threshold: float = 0.75
    min_rubrics: int = 5
    max_rubrics: int = 10


@dataclass
class EvaluationConfig:
    """Configuration for evaluation module."""

    max_concurrent: int = 10
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    timeout_per_rubric: int = 60
    retry_on_parse_failure: bool = True
    score_scale: Tuple[int, int] = field(default_factory=lambda: (1, 5))


@dataclass
class Config:
    """Main configuration container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))
    log_level: str = "INFO"

    def __post_init__(self):
        load_dotenv()
        self.log_level = os.getenv("RETROCHAT_LOG_LEVEL", "INFO")

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        load_dotenv()
        return cls(
            llm=LLMConfig(
                api_key=os.getenv("GOOGLE_API_KEY"),
            ),
            log_level=os.getenv("RETROCHAT_LOG_LEVEL", "INFO"),
        )
