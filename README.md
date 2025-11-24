# Retrochat Evaluator

AI Agent Efficiency Evaluator using LLM-as-a-judge methodology.

## Overview

This tool evaluates "AI Agent Efficiency of User" by:
1. **Training**: Learning evaluation rubrics from high-quality chat sessions
2. **Evaluation**: Scoring new chat sessions against learned rubrics

## Documentation

- [Project Plan](docs/PLAN.md) - Architecture and overview
- [Training Module](docs/PLAN_TRAINING.md) - Detailed training implementation
- [Evaluation Module](docs/PLAN_EVALUATION.md) - Detailed evaluation implementation
- [Prompt Templates](docs/PROMPT_TEMPLATES.md) - LLM prompt specifications
- [Implementation Checklist](docs/IMPLEMENTATION_CHECKLIST.md) - Step-by-step guide

## Tech Stack

- Python 3.11+
- uv (package manager)
- black (formatter)
- LangChain + Gemini 2.5 Pro

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

# Train rubrics from dataset
uv run retrochat-eval train \
    --dataset-dir ./raw-data \
    --dataset-manifest ./dataset.json \
    --output ./output

# Evaluate a chat session
uv run retrochat-eval evaluate \
    --rubrics ./output/train-result-YYYY-MM-DD-HHMMSS/rubrics.json \
    --session ./session.jsonl \
    --output ./result.json
```

## Project Structure

```
retrochat-evaluator/
├── src/retrochat_evaluator/   # Main package
│   ├── training/              # Rubric extraction & summarization
│   ├── evaluation/            # LLM-as-a-judge scoring
│   ├── models/                # Data models (Pydantic)
│   └── llm/                   # Gemini client
├── prompts/                   # Prompt templates
├── raw-data/                  # Example chat sessions
├── output/                    # Training results (gitignored)
│   └── train-result-YYYY-MM-DD-HHMMSS/
│       ├── rubrics.json       # Final consolidated rubrics
│       ├── metadata.json      # Training config and metadata
│       └── raw-rubrics.json    # Session-to-rubrics mapping
└── docs/                      # Documentation
```

## Training Output Structure

Each training run creates a timestamped folder under `output/`:
- `rubrics.json`: Final consolidated rubrics (for validation/evaluation)
- `metadata.json`: Training configuration, LLM settings, and metadata (evaluations can be added later)
- `raw-rubrics.json`: Mapping of session IDs to their extracted rubrics (before consolidation)

## License

MIT
