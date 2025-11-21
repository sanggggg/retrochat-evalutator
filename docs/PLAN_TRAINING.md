# Training Module - Detailed Implementation Plan

## Overview

The Training Module extracts and consolidates evaluation rubrics from high-quality chat sessions.

## Input Specification

### 1. Rubric Extractor Prompt
- Location: `prompts/rubric_extractor.txt`
- Purpose: Instruct LLM to extract evaluation rubrics from a single chat session
- Variables: `{chat_session}` - formatted chat session content

### 2. Dataset
- Format: JSON manifest file pointing to JSONL chat sessions with scores
- Example manifest (`dataset.json`):
```json
{
  "sessions": [
    {
      "file": "example1.jsonl",
      "score": 4.5,
      "metadata": {
        "task_type": "code_generation",
        "date": "2025-11-06"
      }
    },
    {
      "file": "example2.jsonl",
      "score": 3.2,
      "metadata": {
        "task_type": "debugging",
        "date": "2025-11-07"
      }
    }
  ]
}
```

### 3. Rubric Summarizer Prompt
- Location: `prompts/rubric_summarizer.txt`
- Purpose: Consolidate multiple rubric lists into a final coherent set
- Variables: `{all_rubrics}` - concatenated rubrics from all sessions

## Processing Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                                                   │
│  │   Dataset   │                                                   │
│  │  Manifest   │                                                   │
│  └──────┬──────┘                                                   │
│         │                                                          │
│         v                                                          │
│  ┌─────────────────────────────────────────────┐                  │
│  │          Filter by Score Threshold           │                  │
│  │     (keep sessions with score >= N)          │                  │
│  └──────────────────────┬──────────────────────┘                  │
│                         │                                          │
│         ┌───────────────┼───────────────┐                         │
│         │               │               │                         │
│         v               v               v                         │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
│    │Session 1│    │Session 2│    │Session N│                     │
│    └────┬────┘    └────┬────┘    └────┬────┘                     │
│         │               │               │                         │
│         v               v               v                         │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
│    │Extract  │    │Extract  │    │Extract  │  (Parallel LLM)     │
│    │Rubrics  │    │Rubrics  │    │Rubrics  │                     │
│    └────┬────┘    └────┬────┘    └────┬────┘                     │
│         │               │               │                         │
│         └───────────────┼───────────────┘                         │
│                         │                                          │
│                         v                                          │
│              ┌─────────────────────┐                              │
│              │  Concatenate All    │                              │
│              │  Extracted Rubrics  │                              │
│              └──────────┬──────────┘                              │
│                         │                                          │
│                         v                                          │
│              ┌─────────────────────┐                              │
│              │  Rubric Summarizer  │  (LLM Call)                  │
│              │  - Deduplicate      │                              │
│              │  - Merge similar    │                              │
│              │  - Prioritize       │                              │
│              └──────────┬──────────┘                              │
│                         │                                          │
│                         v                                          │
│              ┌─────────────────────┐                              │
│              │   Final Rubrics     │                              │
│              │   (JSON output)     │                              │
│              └─────────────────────┘                              │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Dataset Loader (`training/loader.py`)

```python
class DatasetLoader:
    """Load and filter chat sessions from dataset manifest."""

    def __init__(self, dataset_dir: Path, manifest_path: Path):
        self.dataset_dir = dataset_dir
        self.manifest_path = manifest_path

    def load_manifest(self) -> DatasetManifest:
        """Load dataset manifest JSON."""
        pass

    def filter_by_score(self, threshold: float) -> List[ScoredSession]:
        """Return sessions with score >= threshold."""
        pass

    def load_session(self, session_info: SessionInfo) -> ChatSession:
        """Parse JSONL file into ChatSession model."""
        pass
```

### 2. Rubric Extractor (`training/extractor.py`)

```python
class RubricExtractor:
    """Extract rubrics from a single chat session using LLM."""

    def __init__(self, llm_client: GeminiClient, prompt_template: str):
        self.llm = llm_client
        self.prompt_template = prompt_template

    async def extract(self, session: ChatSession) -> List[Rubric]:
        """Extract rubrics from a chat session."""
        formatted_session = self._format_session(session)
        prompt = self.prompt_template.format(chat_session=formatted_session)
        response = await self.llm.generate(prompt)
        return self._parse_rubrics(response)

    async def extract_batch(
        self,
        sessions: List[ChatSession]
    ) -> List[List[Rubric]]:
        """Extract rubrics from multiple sessions in parallel."""
        tasks = [self.extract(session) for session in sessions]
        return await asyncio.gather(*tasks)

    def _format_session(self, session: ChatSession) -> str:
        """Format chat session for prompt insertion."""
        pass

    def _parse_rubrics(self, response: str) -> List[Rubric]:
        """Parse LLM response into Rubric objects."""
        pass
```

### 3. Rubric Summarizer (`training/summarizer.py`)

```python
class RubricSummarizer:
    """Consolidate multiple rubric lists into final set."""

    def __init__(self, llm_client: GeminiClient, prompt_template: str):
        self.llm = llm_client
        self.prompt_template = prompt_template

    async def summarize(
        self,
        rubric_lists: List[List[Rubric]]
    ) -> List[Rubric]:
        """Consolidate rubrics into final list."""
        all_rubrics = self._flatten_and_format(rubric_lists)
        prompt = self.prompt_template.format(all_rubrics=all_rubrics)
        response = await self.llm.generate(prompt)
        return self._parse_final_rubrics(response)

    def _flatten_and_format(
        self,
        rubric_lists: List[List[Rubric]]
    ) -> str:
        """Combine all rubrics into formatted string."""
        pass

    def _parse_final_rubrics(self, response: str) -> List[Rubric]:
        """Parse LLM response into final Rubric list."""
        pass
```

### 4. Training Orchestrator (`training/trainer.py`)

```python
class Trainer:
    """Orchestrate the complete training pipeline."""

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        extractor: RubricExtractor,
        summarizer: RubricSummarizer,
        config: TrainingConfig
    ):
        self.loader = dataset_loader
        self.extractor = extractor
        self.summarizer = summarizer
        self.config = config

    async def train(self) -> List[Rubric]:
        """Execute full training pipeline."""
        # 1. Load and filter sessions
        sessions = self.loader.filter_by_score(self.config.score_threshold)

        # 2. Load chat content for each session
        chat_sessions = [
            self.loader.load_session(s) for s in sessions
        ]

        # 3. Extract rubrics from each session (parallel)
        rubric_lists = await self.extractor.extract_batch(chat_sessions)

        # 4. Summarize into final rubrics
        final_rubrics = await self.summarizer.summarize(rubric_lists)

        return final_rubrics

    def save_rubrics(self, rubrics: List[Rubric], output_path: Path):
        """Save rubrics to JSON file."""
        pass
```

## Output Specification

### Final Rubrics (`rubrics.json`)

```json
{
  "version": "1.0",
  "created_at": "2025-11-21T10:00:00Z",
  "training_config": {
    "score_threshold": 4.0,
    "sessions_used": 15,
    "total_sessions": 25
  },
  "rubrics": [
    {
      "id": "rubric_001",
      "name": "Task Completion Efficiency",
      "description": "Measures how efficiently the user guides the AI to complete tasks",
      "scoring_criteria": "1: Multiple failed attempts with no progress\n2: Task completed with significant back-and-forth\n3: Task completed with moderate guidance\n4: Task completed with minimal guidance\n5: Task completed in optimal manner",
      "weight": 1.5
    },
    {
      "id": "rubric_002",
      "name": "Clear Communication",
      "description": "Evaluates clarity of user instructions and requirements",
      "scoring_criteria": "1: Vague, ambiguous requests\n2: Partially clear but missing key details\n3: Adequately clear requirements\n4: Clear and specific instructions\n5: Exceptionally clear with edge cases covered",
      "weight": 1.0
    }
  ]
}
```

## Chat Session Formatting

For LLM prompt insertion, chat sessions are formatted as:

```
=== Chat Session ===
Session ID: 0fb7d8cd-be55-431c-ac8c-026b6d6e03dd
Task: (extracted from first user message if possible)

--- Turn 1 ---
[USER]
data path 와 sample size 는 config.yaml 이 아니라, 커맨드라인으로만 입력하게 해줘

[ASSISTANT]
I'll help you modify the project to accept data path and sample size only through command line arguments.

[TOOL_CALLS]
- Glob: *.yaml
- Read: config.yaml

--- Turn 2 ---
[USER]
(continues...)
```

## Error Handling

1. **JSONL Parse Errors**: Skip malformed lines, log warning
2. **LLM Rate Limits**: Implement exponential backoff
3. **Empty Responses**: Retry with increased temperature
4. **Insufficient Sessions**: Warn if fewer than 3 sessions pass threshold

## Configuration Options

```python
@dataclass
class TrainingConfig:
    score_threshold: float = 4.0
    max_sessions: Optional[int] = None  # None = use all
    max_concurrent_extractions: int = 5
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
```
