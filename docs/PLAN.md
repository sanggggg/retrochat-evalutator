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
├── .python-version
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
│       │   └── evaluation.py         # Evaluation result models
│       ├── training/
│       │   ├── __init__.py
│       │   ├── extractor.py          # Rubric extraction from sessions
│       │   ├── summarizer.py         # Rubric consolidation
│       │   └── trainer.py            # Training orchestrator
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── judge.py              # LLM-as-a-judge implementation
│       │   └── evaluator.py          # Evaluation orchestrator
│       ├── llm/
│       │   ├── __init__.py
│       │   └── gemini.py             # Gemini client via LangChain
│       └── utils/
│           ├── __init__.py
│           ├── jsonl.py              # JSONL file handling
│           └── prompts.py            # Prompt loading utilities
├── prompts/
│   ├── rubric_extractor.txt
│   ├── rubric_summarizer.txt
│   └── judge_template.txt
├── raw-data/
│   └── *.jsonl
└── tests/
    ├── __init__.py
    ├── test_training.py
    └── test_evaluation.py
```

## Module Details

### 1. Training Module (`src/retrochat_evaluator/training/`)

See [PLAN_TRAINING.md](./PLAN_TRAINING.md) for detailed implementation plan.

**Purpose**: Learn evaluation rubrics from high-quality chat sessions.

**Workflow**:
1. Load dataset with chat sessions and their scores
2. Filter sessions with score >= threshold
3. Extract rubrics from each qualified session using LLM
4. Consolidate all extracted rubrics into final rubric list

### 2. Evaluation Module (`src/retrochat_evaluator/evaluation/`)

See [PLAN_EVALUATION.md](./PLAN_EVALUATION.md) for detailed implementation plan.

**Purpose**: Evaluate a chat session against learned rubrics.

**Workflow**:
1. Load rubrics and chat session
2. Generate judge prompts for each rubric
3. Execute parallel LLM calls for scoring
4. Aggregate and return scores

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
    --dataset-dir ./raw-data \
    --dataset-manifest ./dataset.json \
    --score-threshold 4.0 \
    --output ./rubrics.json

# Evaluation: Evaluate a chat session
uv run retrochat-eval evaluate \
    --rubrics ./rubrics.json \
    --session ./session.jsonl \
    --output ./result.json

# Full pipeline: Train + Evaluate
uv run retrochat-eval run \
    --dataset-dir ./raw-data \
    --dataset-manifest ./dataset.json \
    --test-session ./test_session.jsonl
```

## Implementation Phases

### Phase 1: Project Setup
- [ ] Initialize uv project with pyproject.toml
- [ ] Set up black formatter
- [ ] Create directory structure
- [ ] Add LangChain + Gemini dependencies

### Phase 2: Core Data Models
- [ ] Implement chat session parser
- [ ] Implement rubric data model
- [ ] Implement evaluation result model

### Phase 3: LLM Integration
- [ ] Set up LangChain with Gemini 2.5 Pro
- [ ] Implement prompt loading utilities
- [ ] Create base LLM wrapper

### Phase 4: Training Module
- [ ] Implement rubric extractor
- [ ] Implement rubric summarizer
- [ ] Create training orchestrator

### Phase 5: Evaluation Module
- [ ] Implement judge prompt generator
- [ ] Implement parallel scoring
- [ ] Create evaluation orchestrator

### Phase 6: CLI & Integration
- [ ] Implement CLI commands
- [ ] Add configuration management
- [ ] Integration testing

## Configuration

Environment variables:
- `GOOGLE_API_KEY`: Gemini API key
- `RETROCHAT_LOG_LEVEL`: Logging level (default: INFO)

## Dependencies

```toml
[project]
dependencies = [
    "langchain>=0.3.0",
    "langchain-google-genai>=2.0.0",
    "pydantic>=2.0.0",
    "click>=8.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "black>=24.0.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]
```
