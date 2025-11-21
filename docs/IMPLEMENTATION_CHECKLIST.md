# Implementation Checklist

Step-by-step implementation guide with dependencies.

## Phase 1: Project Setup

### 1.1 Initialize Project Structure
- [ ] Create `pyproject.toml` with uv configuration
- [ ] Create `.python-version` file (3.11+)
- [ ] Create `.gitignore` for Python projects
- [ ] Set up directory structure:
  ```
  src/retrochat_evaluator/
  prompts/
  tests/
  docs/
  ```

### 1.2 Configure Dependencies
```toml
# pyproject.toml
[project]
name = "retrochat-evaluator"
version = "0.1.0"
requires-python = ">=3.11"
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

[project.scripts]
retrochat-eval = "retrochat_evaluator.cli:main"
```

### 1.3 Configure Development Tools
- [ ] Add `pyproject.toml` black configuration
- [ ] Create `.env.example` template
- [ ] Initialize uv: `uv init` or create project manually

---

## Phase 2: Core Data Models

### 2.1 Rubric Model (`src/retrochat_evaluator/models/rubric.py`)
- [ ] Define `Rubric` Pydantic model
- [ ] Define `RubricList` wrapper with serialization
- [ ] Add validation for scoring_criteria format
- [ ] Add JSON import/export methods

### 2.2 Chat Session Model (`src/retrochat_evaluator/models/chat_session.py`)
- [ ] Define `ChatMessage` model (user/assistant)
- [ ] Define `ToolCall` model
- [ ] Define `ChatSession` model
- [ ] Implement JSONL parser for Claude Code format
- [ ] Add session formatter for prompts

### 2.3 Evaluation Result Model (`src/retrochat_evaluator/models/evaluation.py`)
- [ ] Define `RubricScore` model
- [ ] Define `EvaluationResult` model
- [ ] Define `BatchEvaluationSummary` model
- [ ] Add JSON export methods

---

## Phase 3: LLM Integration

### 3.1 Gemini Client (`src/retrochat_evaluator/llm/gemini.py`)
- [ ] Set up LangChain ChatGoogleGenerativeAI
- [ ] Create async wrapper for batch calls
- [ ] Implement rate limiting / retry logic
- [ ] Add configuration for temperature, max_tokens

### 3.2 Prompt Utilities (`src/retrochat_evaluator/utils/prompts.py`)
- [ ] Implement prompt template loader
- [ ] Add variable substitution
- [ ] Validate required variables are provided

### 3.3 JSONL Utilities (`src/retrochat_evaluator/utils/jsonl.py`)
- [ ] Implement JSONL file reader
- [ ] Add streaming parser for large files
- [ ] Handle malformed lines gracefully

---

## Phase 4: Training Module

### 4.1 Dataset Loader (`src/retrochat_evaluator/training/loader.py`)
- [ ] Load dataset manifest JSON
- [ ] Filter sessions by score threshold
- [ ] Load individual JSONL sessions
- [ ] Validate dataset structure

### 4.2 Rubric Extractor (`src/retrochat_evaluator/training/extractor.py`)
- [ ] Load extractor prompt template
- [ ] Format chat session for prompt
- [ ] Call LLM for rubric extraction
- [ ] Parse JSON response into Rubric models
- [ ] Implement batch extraction with parallelism

### 4.3 Rubric Summarizer (`src/retrochat_evaluator/training/summarizer.py`)
- [ ] Load summarizer prompt template
- [ ] Format all extracted rubrics
- [ ] Call LLM for consolidation
- [ ] Parse response into final Rubric list
- [ ] Assign sequential IDs

### 4.4 Training Orchestrator (`src/retrochat_evaluator/training/trainer.py`)
- [ ] Wire together loader, extractor, summarizer
- [ ] Implement `train()` async method
- [ ] Add progress logging
- [ ] Save final rubrics to JSON

---

## Phase 5: Evaluation Module

### 5.1 Judge Prompt Generator (`src/retrochat_evaluator/evaluation/judge.py`)
- [ ] Load judge template prompt
- [ ] Generate prompt per rubric
- [ ] Format chat session consistently

### 5.2 Rubric Scorer (`src/retrochat_evaluator/evaluation/scorer.py`)
- [ ] Call LLM with judge prompt
- [ ] Parse SCORE and REASONING from response
- [ ] Handle parse failures with retry
- [ ] Implement parallel scoring with semaphore

### 5.3 Result Aggregator (`src/retrochat_evaluator/evaluation/aggregator.py`)
- [ ] Calculate weighted total score
- [ ] Build EvaluationResult
- [ ] Generate batch summary statistics

### 5.4 Evaluation Orchestrator (`src/retrochat_evaluator/evaluation/evaluator.py`)
- [ ] Wire together generator, scorer, aggregator
- [ ] Implement `evaluate()` async method
- [ ] Implement `evaluate_batch()` for multiple sessions
- [ ] Save results to JSON

---

## Phase 6: CLI & Integration

### 6.1 CLI Commands (`src/retrochat_evaluator/cli.py`)
- [ ] Set up Click CLI structure
- [ ] Implement `train` command
  - `--dataset-dir`, `--dataset-manifest`, `--score-threshold`, `--output`
- [ ] Implement `evaluate` command
  - `--rubrics`, `--session`, `--output`
- [ ] Implement `evaluate-batch` command
  - `--rubrics`, `--sessions-dir`, `--output-dir`
- [ ] Add `--verbose` flag for debug output
- [ ] Load environment variables from .env

### 6.2 Configuration (`src/retrochat_evaluator/config.py`)
- [ ] Define TrainingConfig dataclass
- [ ] Define EvaluationConfig dataclass
- [ ] Support config via CLI args and env vars

### 6.3 Prompt Templates
- [ ] Create `prompts/rubric_extractor.txt`
- [ ] Create `prompts/rubric_summarizer.txt`
- [ ] Create `prompts/judge_template.txt`

---

## Phase 7: Testing

### 7.1 Unit Tests
- [ ] Test JSONL parser with sample data
- [ ] Test Rubric model serialization
- [ ] Test chat session formatter
- [ ] Test prompt template loading

### 7.2 Integration Tests
- [ ] Test rubric extraction with mock LLM
- [ ] Test rubric summarization with mock LLM
- [ ] Test evaluation scoring with mock LLM

### 7.3 End-to-End Tests
- [ ] Test full training pipeline with example data
- [ ] Test full evaluation pipeline

---

## Dependencies Between Phases

```
Phase 1 (Setup)
    │
    v
Phase 2 (Data Models) ──────────────────┐
    │                                   │
    v                                   │
Phase 3 (LLM Integration)               │
    │                                   │
    ├───────────────┬───────────────────┤
    │               │                   │
    v               v                   │
Phase 4         Phase 5                 │
(Training)      (Evaluation)            │
    │               │                   │
    └───────┬───────┘                   │
            │                           │
            v                           │
      Phase 6 (CLI) <───────────────────┘
            │
            v
      Phase 7 (Testing)
```

---

## Estimated Time

| Phase | Estimated Time |
|-------|----------------|
| Phase 1: Setup | 30 min |
| Phase 2: Data Models | 2 hours |
| Phase 3: LLM Integration | 1.5 hours |
| Phase 4: Training Module | 3 hours |
| Phase 5: Evaluation Module | 2.5 hours |
| Phase 6: CLI & Integration | 1.5 hours |
| Phase 7: Testing | 2 hours |
| **Total** | **~13 hours** |
