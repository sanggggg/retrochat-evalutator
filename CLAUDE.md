# CLAUDE.md - Project Guide for Claude Code

## Project Overview

**retrochat-evaluator** is an AI Agent Efficiency Evaluator that uses LLM-as-a-judge methodology to evaluate how efficiently users interact with AI coding assistants.

### Core Concepts

1. **Training Module**: Extracts evaluation rubrics from high-quality chat sessions
2. **Evaluation Module**: Scores new chat sessions against learned rubrics using LLM-as-a-judge

## Tech Stack

- **Python 3.11+** with type hints
- **uv** for package management
- **LangChain + Google Gemini** for LLM integration
- **Pydantic** for data models
- **Click** for CLI
- **pytest + pytest-asyncio** for testing

## Project Structure

```
retrochat-evaluator/
├── src/retrochat_evaluator/
│   ├── models/              # Pydantic data models
│   │   ├── rubric.py        # Rubric, RubricList
│   │   ├── chat_session.py  # ChatSession, ChatMessage, ToolCall
│   │   └── evaluation.py    # RubricScore, EvaluationResult
│   ├── training/            # Training pipeline
│   │   ├── loader.py        # Dataset loading & filtering
│   │   ├── extractor.py     # Rubric extraction from sessions
│   │   ├── summarizer.py    # Rubric consolidation
│   │   └── trainer.py       # Training orchestrator
│   ├── evaluation/          # Evaluation pipeline
│   │   ├── judge.py         # Judge prompt generator
│   │   ├── scorer.py        # LLM-as-a-judge scoring
│   │   ├── aggregator.py    # Result aggregation
│   │   └── evaluator.py     # Evaluation orchestrator
│   ├── llm/
│   │   └── gemini.py        # Gemini client via LangChain
│   ├── utils/
│   │   ├── prompts.py       # Prompt loading utilities
│   │   └── jsonl.py         # JSONL file handling
│   ├── config.py            # Configuration dataclasses
│   └── cli.py               # Click CLI interface
├── prompts/                 # LLM prompt templates
│   ├── rubric_extractor.txt
│   ├── rubric_summarizer.txt
│   └── judge_template.txt
├── tests/                   # Test suite
│   ├── fixtures/            # Mock data files
│   └── test_*.py            # Test modules
├── raw-data/                # Example JSONL chat sessions
└── docs/                    # Documentation
```

## Common Commands

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_models.py -v

# Format code
uv run black src/ tests/

# Train rubrics from dataset
uv run retrochat-eval train \
    --dataset-dir ./raw-data \
    --dataset-manifest ./dataset.json \
    --output ./rubrics.json

# Evaluate a session
uv run retrochat-eval evaluate \
    --rubrics ./rubrics.json \
    --session ./session.jsonl \
    --output ./result.json

# Batch evaluate
uv run retrochat-eval evaluate-batch \
    --rubrics ./rubrics.json \
    --sessions-dir ./sessions/ \
    --output-dir ./results/
```

## Key Patterns

### Async Operations
All LLM calls use async/await. Use `asyncio.run()` in CLI commands:
```python
result = asyncio.run(evaluator.evaluate(rubrics, session))
```

### Pydantic Models
All data models inherit from `pydantic.BaseModel` with JSON serialization:
```python
rubric_list.to_json(path)
RubricList.from_json(path)
```

### Prompt Templates
Prompts use simple `{variable}` substitution loaded from `prompts/` directory:
```python
template = load_prompt_template(Path("rubric_extractor.txt"), prompts_dir)
prompt = format_prompt(template, chat_session=formatted)
```

### Chat Session Parsing
Chat sessions are parsed from Claude Code JSONL format with types:
- `file-history-snapshot`: Skipped
- `user`: User messages
- `assistant`: Assistant responses with tool calls

## Environment Variables

```bash
GOOGLE_API_KEY=your_gemini_api_key  # Required for LLM calls
RETROCHAT_LOG_LEVEL=INFO            # Optional logging level
```

## Testing Guidelines

- Tests use **mock LLM responses** - no real API calls
- Fixtures are in `tests/conftest.py`
- Mock chat sessions are in `tests/fixtures/`
- Use `@pytest.mark.asyncio` for async tests (auto mode enabled)

### Running Tests
```bash
# All tests
uv run pytest tests/ -v

# With coverage (if installed)
uv run pytest tests/ -v --cov=retrochat_evaluator
```

## Data Formats

### Dataset Manifest (dataset.json)
```json
{
  "sessions": [
    {"file": "session1.jsonl", "score": 4.5, "metadata": {...}},
    {"file": "session2.jsonl", "score": 3.2, "metadata": {...}}
  ]
}
```

### Rubrics Output (rubrics.json)
```json
{
  "version": "1.0",
  "created_at": "2025-11-21T10:00:00Z",
  "training_config": {...},
  "rubrics": [
    {"id": "rubric_001", "name": "...", "description": "...", "scoring_criteria": "...", "weight": 1.0}
  ]
}
```

### Evaluation Result (result.json)
```json
{
  "session_id": "...",
  "rubric_scores": [
    {"rubric_id": "rubric_001", "score": 4.0, "reasoning": "..."}
  ],
  "summary": {"total_score": 4.2, "percentage": 84.0}
}
```

## Architecture Notes

### Training Pipeline Flow
1. `DatasetLoader` loads manifest and filters by score threshold
2. `RubricExtractor` extracts rubrics from each qualified session (parallel)
3. `RubricSummarizer` consolidates all rubrics into final set
4. `Trainer` orchestrates the pipeline

### Evaluation Pipeline Flow
1. `JudgePromptGenerator` creates prompts for each rubric
2. `RubricScorer` calls LLM for each rubric (parallel with semaphore)
3. `ResultAggregator` calculates weighted scores
4. `Evaluator` orchestrates the pipeline

## Code Style

- Use **black** formatter with 100 char line length
- Type hints required for all function signatures
- Docstrings for public classes and methods
- Async functions for any I/O or LLM operations
