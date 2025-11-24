# AI Agent Efficiency Evaluator - Project Plan

## Overview

This project evaluates "AI Agent Efficiency of User" using an LLM-as-a-judge methodology. The system learns evaluation rubrics from high-quality chat sessions and applies them to evaluate new sessions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Module                               │
│  ┌──────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │ Dataset  │──>│ Rubric Extractor│──>│ Rubric Summarizer    │  │
│  │ (scored) │   │ (per session)   │   │ (consolidate all)    │  │
│  └──────────┘   └─────────────────┘   └──────────────────────┘  │
│                                              │                   │
│                                              v                   │
│                                    ┌─────────────────┐          │
│                                    │ Final Rubrics   │          │
│                                    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation Module                             │
│  ┌──────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │ Rubrics  │──>│ Judge Prompt    │──>│ Per-Rubric Scoring   │  │
│  │          │   │ Generator       │   │ (parallel LLM calls) │  │
│  └──────────┘   └─────────────────┘   └──────────────────────┘  │
│                                              │                   │
│        ┌──────────────┐                      v                   │
│        │ Chat Session │──────────>  ┌─────────────────┐         │
│        └──────────────┘             │ Score per Rubric│         │
│                                     └─────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Language**: Python 3.11+
- **Package Manager**: uv
- **Code Formatter**: black
- **LLM Provider**: Google Gemini 2.5 Pro
- **LLM Framework**: LangChain

## Project Structure

```
retrochat-evaluator/
├── pyproject.toml
├── config.yaml                    # YAML configuration file (optional)
├── config.example.yaml            # Example configuration
├── dataset.json                   # Dataset manifest
├── README.md
├── docs/
│   ├── PLAN.md
│   └── PROMPT_TEMPLATES.md
├── src/
│   └── retrochat_evaluator/
│       ├── __init__.py
│       ├── cli.py                    # CLI entry points
│       ├── config.py                 # Configuration management
│       ├── models/
│       │   ├── __init__.py
│       │   ├── rubric.py             # Rubric data models
│       │   ├── chat_session.py       # Chat session data models
│       │   ├── evaluation.py         # Evaluation result models
│       │   └── validation.py         # Validation result models
│       ├── training/
│       │   ├── __init__.py
│       │   ├── extractor.py          # Rubric extraction from sessions
│       │   ├── summarizer.py         # LLM-based rubric consolidation
│       │   ├── semantic_summarizer.py # Embedding-based rubric consolidation
│       │   ├── visualizer.py         # Clustering visualization
│       │   ├── loader.py             # Dataset loading
│       │   └── trainer.py            # Training orchestrator
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── judge.py              # LLM-as-a-judge prompt generation
│       │   ├── scorer.py             # Parallel scoring
│       │   ├── aggregator.py         # Score aggregation
│       │   └── evaluator.py          # Evaluation orchestrator
│       ├── validation/
│       │   ├── __init__.py
│       │   └── validator.py          # Validation pipeline
│       ├── llm/
│       │   ├── __init__.py
│       │   └── gemini.py             # Gemini client via LangChain
│       └── utils/
│           ├── __init__.py
│           ├── jsonl.py              # JSONL file handling
│           └── prompts.py            # Prompt loading utilities
├── prompts/                   # Prompt templates (see PROMPT_TEMPLATES.md)
├── input/                          # Input directory for sessions
│   └── *.jsonl
├── output/                         # Training results (gitignored)
│   └── train-result-YYYY-MM-DD-HHMMSS/
│       ├── rubrics.json
│       ├── raw-rubrics.json
│       ├── metadata.json
│       └── clustering_visualization.png (optional)
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── fixtures/
    │   ├── mock_manifest.json
    │   ├── mock_session.jsonl
    │   └── mock_session_2.jsonl
    ├── test_training.py
    ├── test_evaluation.py
    ├── test_models.py
    ├── test_utils.py
    └── test_validation.py
```

## Module Details

### 1. Training Module (`src/retrochat_evaluator/training/`)

**Purpose**: Learn evaluation rubrics from high-quality chat sessions.

**Workflow**:
1. Load dataset with chat sessions and their scores
2. Filter sessions with score >= threshold (by score name)
3. Extract rubrics from each qualified session using LLM
4. Consolidate all extracted rubrics into final rubric list (using LLM or semantic clustering)
5. Automatically validate trained rubrics on validation split

**Components**:
- `loader.py`: Dataset loading and filtering
- `extractor.py`: Rubric extraction from sessions (supports multiple extractor prompts based on score type)
- `summarizer.py`: LLM-based rubric consolidation
- `semantic_summarizer.py`: Embedding-based rubric consolidation using HAC clustering
- `visualizer.py`: Clustering visualization
- `trainer.py`: Training orchestrator

### 2. Evaluation Module (`src/retrochat_evaluator/evaluation/`)

**Purpose**: Evaluate a chat session against learned rubrics.

**Workflow**:
1. Load rubrics and chat session
2. Generate judge prompts for each rubric
3. Execute parallel LLM calls for scoring
4. Aggregate and return scores

**Components**:
- `judge.py`: Judge prompt generation
- `scorer.py`: Parallel LLM scoring
- `aggregator.py`: Score aggregation
- `evaluator.py`: Evaluation orchestrator

## Data Models

### Chat Session (from JSONL)

Based on the example data, each line in JSONL can be:
- `file-history-snapshot`: File state snapshot
- `user`: User message with content
- `assistant`: Assistant response with tool calls/text

### Rubric

```python
@dataclass
class Rubric:
    id: str
    name: str
    description: str
    scoring_criteria: str  # How to score 1-5 or pass/fail
    weight: float = 1.0
```

### Evaluation Result

```python
@dataclass
class RubricScore:
    rubric_id: str
    score: float
    reasoning: str

@dataclass
class EvaluationResult:
    session_id: str
    rubric_scores: List[RubricScore]
    total_score: float
```

## CLI Interface

```bash
# Training: Generate rubrics from dataset
uv run retrochat-eval train \
    --dataset-dir ./input \
    --dataset-manifest ./dataset.json \
    --score-threshold 4.0 \
    --score-name default \
    --output ./output

# Training with YAML config
uv run retrochat-eval train --config ./config.yaml

# Training with semantic clustering summarization
uv run retrochat-eval train \
    --dataset-dir ./input \
    --dataset-manifest ./dataset.json \
    --summarization-method semantic_clustering \
    --similarity-threshold 0.75

# Evaluation: Evaluate a chat session
uv run retrochat-eval evaluate \
    --rubrics ./output/train-result-YYYY-MM-DD-HHMMSS/rubrics.json \
    --session ./session.jsonl \
    --output ./result.json

# Batch evaluation: Evaluate multiple sessions
uv run retrochat-eval evaluate-batch \
    --rubrics ./output/train-result-YYYY-MM-DD-HHMMSS/rubrics.json \
    --sessions-dir ./sessions/ \
    --output-dir ./results/

# Validation: Validate rubrics against dataset
uv run retrochat-eval validate \
    --dataset-dir ./input \
    --dataset-manifest ./dataset.json \
    --rubrics ./output/train-result-YYYY-MM-DD-HHMMSS/rubrics.json \
    --output ./validation_report.json
```

## Implementation Status

All core modules have been implemented:

- ✅ **Project Setup**: uv project with pyproject.toml, black formatter, directory structure
- ✅ **Core Data Models**: Chat session parser, rubric data model, evaluation result model
- ✅ **LLM Integration**: LangChain with Gemini 2.5 Pro, prompt loading utilities, LLM wrapper
- ✅ **Training Module**: Rubric extractor (with multiple variants), LLM-based and semantic clustering summarizers, training orchestrator, clustering visualization
- ✅ **Evaluation Module**: Judge prompt generator, parallel scoring, score aggregator, evaluation orchestrator
- ✅ **Validation Module**: Validation pipeline, predicted vs real score comparison, validation metrics (MAE, RMSE, correlation)
- ✅ **CLI & Integration**: CLI commands (train, evaluate, evaluate-batch, validate), YAML configuration support, environment variable support, integration testing

## Configuration

Configuration can be provided via:
1. **YAML config file** (recommended): `--config ./config.yaml`
2. **CLI arguments**: Override config file values
3. **Environment variables**: Fallback defaults

### Environment Variables
- `GOOGLE_API_KEY`: Gemini API key (required)
- `RETROCHAT_LOG_LEVEL`: Logging level (default: INFO)

### YAML Config File
See `config.example.yaml` for full configuration options including:
- LLM settings (model, temperature, max_tokens) for extraction, summarization, and evaluation
- Training settings (score threshold, score name, summarization method)
- Path configurations (dataset directory, manifest, prompts directory)
- Semantic clustering settings (embedding model, similarity threshold)

## Dependencies

```toml
[project]
dependencies = [
    "langchain>=0.3.0",
    "langchain-google-genai>=2.0.0",
    "pydantic>=2.0.0",
    "click>=8.0.0",
    "python-dotenv>=1.0.0",
    "scikit-learn>=1.5.0",      # For semantic clustering
    "numpy>=1.26.0",             # For clustering algorithms
    "pyyaml>=6.0.0",             # For YAML config support
    "matplotlib>=3.8.0",         # For clustering visualization
]

[project.optional-dependencies]
dev = [
    "black>=24.0.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]
```
