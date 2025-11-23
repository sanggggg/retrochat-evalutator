"""Configuration management for Retrochat Evaluator."""

import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Tuple, Any
from pathlib import Path

import yaml
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
class ExtractionLLMConfig(LLMConfig):
    """LLM configuration for rubric extraction."""

    model_name: str = "gemini-2.5-pro-preview-06-05"
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class SummarizationLLMConfig(LLMConfig):
    """LLM configuration for rubric summarization."""

    model_name: str = "gemini-2.5-pro-preview-06-05"
    temperature: float = 0.3
    max_tokens: int = 8192


@dataclass
class EvaluationLLMConfig(LLMConfig):
    """LLM configuration for evaluation/scoring."""

    model_name: str = "gemini-2.5-pro-preview-06-05"
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass
class TrainingConfig:
    """Configuration for training module."""

    score_threshold: float = 4.0
    score_name: str = "default"
    max_sessions: Optional[int] = None
    max_concurrent_extractions: int = 5

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
    timeout_per_rubric: int = 60
    retry_on_parse_failure: bool = True
    score_scale: Tuple[int, int] = field(default_factory=lambda: (1, 5))


@dataclass
class Config:
    """Main configuration container."""

    # Task-specific LLM configurations
    extraction_llm: ExtractionLLMConfig = field(default_factory=ExtractionLLMConfig)
    summarization_llm: SummarizationLLMConfig = field(default_factory=SummarizationLLMConfig)
    evaluation_llm: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))
    log_level: str = "INFO"

    # Path configurations
    dataset_dir: Optional[Path] = None
    dataset_manifest: Optional[Path] = None
    rubrics_path: Optional[Path] = None
    output_path: Optional[Path] = None

    def __post_init__(self):
        load_dotenv()
        self.log_level = os.getenv("RETROCHAT_LOG_LEVEL", "INFO")

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        load_dotenv()
        return cls(
            log_level=os.getenv("RETROCHAT_LOG_LEVEL", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        data = asdict(self)
        # Convert Path to string
        data["prompts_dir"] = str(self.prompts_dir)
        # Convert optional paths to string or remove if None
        for key in ["dataset_dir", "dataset_manifest", "rubrics_path", "output_path"]:
            if data[key] is not None:
                data[key] = str(data[key])
            else:
                del data[key]
        # Convert Enum to string
        data["training"]["summarization_method"] = self.training.summarization_method.value
        # Convert tuple to list for YAML compatibility
        data["evaluation"]["score_scale"] = list(self.evaluation.score_scale)
        # Remove api_key from LLM configs
        for llm_key in ["extraction_llm", "summarization_llm", "evaluation_llm"]:
            data[llm_key].pop("api_key", None)
        return data

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        extraction_llm_data = data.get("extraction_llm", {})
        summarization_llm_data = data.get("summarization_llm", {})
        evaluation_llm_data = data.get("evaluation_llm", {})
        training_data = data.get("training", {})
        evaluation_data = data.get("evaluation", {})

        # Handle summarization_method enum
        if "summarization_method" in training_data:
            training_data["summarization_method"] = SummarizationMethod(
                training_data["summarization_method"]
            )

        # Handle score_scale tuple
        if "score_scale" in evaluation_data:
            evaluation_data["score_scale"] = tuple(evaluation_data["score_scale"])

        # Handle prompts_dir Path
        prompts_dir = Path(data.get("prompts_dir", "prompts"))

        # Handle optional path configs
        dataset_dir = Path(data["dataset_dir"]) if data.get("dataset_dir") else None
        dataset_manifest = Path(data["dataset_manifest"]) if data.get("dataset_manifest") else None
        rubrics_path = Path(data["rubrics_path"]) if data.get("rubrics_path") else None
        output_path = Path(data["output_path"]) if data.get("output_path") else None

        return cls(
            extraction_llm=ExtractionLLMConfig(**extraction_llm_data),
            summarization_llm=SummarizationLLMConfig(**summarization_llm_data),
            evaluation_llm=EvaluationLLMConfig(**evaluation_llm_data),
            training=TrainingConfig(**training_data),
            evaluation=EvaluationConfig(**evaluation_data),
            prompts_dir=prompts_dir,
            log_level=data.get("log_level", "INFO"),
            dataset_dir=dataset_dir,
            dataset_manifest=dataset_manifest,
            rubrics_path=rubrics_path,
            output_path=output_path,
        )
