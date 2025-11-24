# Retrochat Evaluator

AI Agent Efficiency Evaluator using LLM-as-a-judge methodology.

## Overview

This tool evaluates "AI Agent Efficiency of User" by:
1. **Training**: Learning evaluation rubrics from high-quality chat sessions
2. **Evaluation**: Scoring new chat sessions against learned rubrics

## Documentation

- [Project Plan](docs/PLAN.md) - Architecture and overview
- [Prompt Templates](docs/PROMPT_TEMPLATES.md) - LLM prompt specifications

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

# Generate dataset manifest (with training/validation split)
python src/scripts/generate_manifest.py \
    --input-dir ./input \
    --output ./dataset.json

# Train rubrics from dataset
# - Uses only training split (90% of data)
# - Automatically validates trained rubrics on validation split (10% of data)
# - Validation results are saved in metadata.json
uv run retrochat-eval train \
    --dataset-dir ./input \
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

## Dataset Split

The dataset is automatically split into training (90%) and validation (10%) sets when generating the manifest:
- **Training set**: Used for extracting and learning rubrics
- **Validation set**: Used for validating trained rubrics (automatically performed after training)

The split is done with a fixed random seed (42) for reproducibility. Each session in `dataset.json` includes a `split` field indicating whether it belongs to "training" or "validation".

## Training Output Structure

Each training run creates a timestamped folder under `output/`:
- `rubrics.json`: Final consolidated rubrics (for validation/evaluation)
- `metadata.json`: Training configuration, LLM settings, and validation results
  - Includes validation metrics (correlation, MAE, RMSE, etc.) comparing predicted vs real scores
  - Validation is automatically performed after training completes
- `raw-rubrics.json`: Mapping of session IDs to their extracted rubrics (before consolidation)
- `clustering_visualization.png`: (Optional) Visualization of semantic clustering if used

## Training Process

1. **Dataset Loading**: Loads sessions from the training split (90% of data)
2. **Rubric Extraction**: Extracts evaluation rubrics from high-scoring sessions
3. **Rubric Consolidation**: Summarizes extracted rubrics into final evaluation criteria
4. **Automatic Validation**: Validates trained rubrics on the validation split (10% of data)
   - Compares LLM-predicted scores against real scores
   - Calculates correlation, MAE, RMSE, and other metrics
   - Results are saved in `metadata.json`

## License

MIT
