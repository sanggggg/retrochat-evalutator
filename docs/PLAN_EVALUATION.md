# Evaluation Module - Detailed Implementation Plan

## Overview

The Evaluation Module scores chat sessions against learned rubrics using an LLM-as-a-judge approach.

## Input Specification

### 1. List of Rubrics
- Source: Output from Training Module or pre-defined rubrics
- Format: JSON file matching the schema from training output
- Location: User-specified via CLI

### 2. LLM-as-a-Judge Template Prompt
- Location: `prompts/judge_template.txt`
- Purpose: Template that gets instantiated per rubric
- Variables:
  - `{rubric_name}`: Name of the rubric being evaluated
  - `{rubric_description}`: Full description of the rubric
  - `{scoring_criteria}`: How to assign scores
  - `{chat_session}`: Formatted chat session content

### 3. Chat Session
- Format: JSONL file (same format as training data)
- Single session to be evaluated

## Processing Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐         ┌──────────────┐                          │
│  │   Rubrics   │         │ Chat Session │                          │
│  │   (JSON)    │         │   (JSONL)    │                          │
│  └──────┬──────┘         └──────┬───────┘                          │
│         │                       │                                   │
│         v                       v                                   │
│  ┌─────────────────────────────────────────────┐                   │
│  │           Load and Parse Inputs              │                   │
│  └──────────────────────┬──────────────────────┘                   │
│                         │                                          │
│         ┌───────────────┼───────────────┐                         │
│         │               │               │                         │
│         v               v               v                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                   │
│  │  Rubric 1  │  │  Rubric 2  │  │  Rubric N  │                   │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                   │
│        │               │               │                          │
│        v               v               v                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                   │
│  │  Generate  │  │  Generate  │  │  Generate  │                   │
│  │Judge Prompt│  │Judge Prompt│  │Judge Prompt│                   │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                   │
│        │               │               │                          │
│        v               v               v                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                   │
│  │  LLM Call  │  │  LLM Call  │  │  LLM Call  │  (Parallel)       │
│  │  (Score)   │  │  (Score)   │  │  (Score)   │                   │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                   │
│        │               │               │                          │
│        └───────────────┼───────────────┘                         │
│                        │                                          │
│                        v                                          │
│           ┌────────────────────────┐                              │
│           │  Aggregate Results     │                              │
│           │  - Per-rubric scores   │                              │
│           │  - Weighted total      │                              │
│           │  - Reasoning           │                              │
│           └───────────┬────────────┘                              │
│                       │                                           │
│                       v                                           │
│           ┌────────────────────────┐                              │
│           │   Evaluation Result    │                              │
│           │       (JSON)           │                              │
│           └────────────────────────┘                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Judge Prompt Generator (`evaluation/judge.py`)

```python
class JudgePromptGenerator:
    """Generate evaluation prompts from template and rubrics."""

    def __init__(self, template_path: Path):
        self.template = self._load_template(template_path)

    def generate(
        self,
        rubric: Rubric,
        session: ChatSession
    ) -> str:
        """Generate a judge prompt for a specific rubric."""
        formatted_session = self._format_session(session)
        return self.template.format(
            rubric_name=rubric.name,
            rubric_description=rubric.description,
            scoring_criteria=rubric.scoring_criteria,
            chat_session=formatted_session
        )

    def generate_all(
        self,
        rubrics: List[Rubric],
        session: ChatSession
    ) -> List[Tuple[Rubric, str]]:
        """Generate judge prompts for all rubrics."""
        return [
            (rubric, self.generate(rubric, session))
            for rubric in rubrics
        ]

    def _load_template(self, path: Path) -> str:
        """Load prompt template from file."""
        pass

    def _format_session(self, session: ChatSession) -> str:
        """Format chat session for prompt insertion."""
        pass
```

### 2. Rubric Scorer (`evaluation/scorer.py`)

```python
class RubricScorer:
    """Score a chat session against rubrics using LLM."""

    def __init__(
        self,
        llm_client: GeminiClient,
        config: EvaluationConfig
    ):
        self.llm = llm_client
        self.config = config

    async def score(
        self,
        rubric: Rubric,
        prompt: str
    ) -> RubricScore:
        """Score a session against a single rubric."""
        response = await self.llm.generate(
            prompt,
            temperature=self.config.llm_temperature
        )
        return self._parse_score(rubric, response)

    async def score_all(
        self,
        rubric_prompts: List[Tuple[Rubric, str]]
    ) -> List[RubricScore]:
        """Score against all rubrics in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def bounded_score(rubric: Rubric, prompt: str):
            async with semaphore:
                return await self.score(rubric, prompt)

        tasks = [
            bounded_score(rubric, prompt)
            for rubric, prompt in rubric_prompts
        ]
        return await asyncio.gather(*tasks)

    def _parse_score(
        self,
        rubric: Rubric,
        response: str
    ) -> RubricScore:
        """Parse LLM response into RubricScore."""
        # Expected format from LLM:
        # SCORE: 4
        # REASONING: The user demonstrated clear communication...
        pass
```

### 3. Result Aggregator (`evaluation/aggregator.py`)

```python
class ResultAggregator:
    """Aggregate individual rubric scores into final result."""

    def aggregate(
        self,
        session_id: str,
        rubric_scores: List[RubricScore],
        rubrics: List[Rubric]
    ) -> EvaluationResult:
        """Aggregate scores into final evaluation result."""
        # Create rubric lookup for weights
        rubric_map = {r.id: r for r in rubrics}

        # Calculate weighted total
        total_weight = sum(rubric_map[s.rubric_id].weight for s in rubric_scores)
        weighted_sum = sum(
            s.score * rubric_map[s.rubric_id].weight
            for s in rubric_scores
        )
        total_score = weighted_sum / total_weight if total_weight > 0 else 0

        return EvaluationResult(
            session_id=session_id,
            rubric_scores=rubric_scores,
            total_score=total_score,
            max_score=5.0,  # Assuming 1-5 scale
            evaluated_at=datetime.utcnow()
        )
```

### 4. Evaluation Orchestrator (`evaluation/evaluator.py`)

```python
class Evaluator:
    """Orchestrate the complete evaluation pipeline."""

    def __init__(
        self,
        prompt_generator: JudgePromptGenerator,
        scorer: RubricScorer,
        aggregator: ResultAggregator,
        config: EvaluationConfig
    ):
        self.prompt_generator = prompt_generator
        self.scorer = scorer
        self.aggregator = aggregator
        self.config = config

    async def evaluate(
        self,
        rubrics: List[Rubric],
        session: ChatSession
    ) -> EvaluationResult:
        """Execute full evaluation pipeline."""
        # 1. Generate judge prompts for each rubric
        rubric_prompts = self.prompt_generator.generate_all(
            rubrics, session
        )

        # 2. Score against all rubrics (parallel LLM calls)
        rubric_scores = await self.scorer.score_all(rubric_prompts)

        # 3. Aggregate into final result
        result = self.aggregator.aggregate(
            session_id=session.session_id,
            rubric_scores=rubric_scores,
            rubrics=rubrics
        )

        return result

    async def evaluate_batch(
        self,
        rubrics: List[Rubric],
        sessions: List[ChatSession]
    ) -> List[EvaluationResult]:
        """Evaluate multiple sessions."""
        return [
            await self.evaluate(rubrics, session)
            for session in sessions
        ]

    def save_result(
        self,
        result: EvaluationResult,
        output_path: Path
    ):
        """Save evaluation result to JSON."""
        pass
```

## Output Specification

### Evaluation Result (`result.json`)

```json
{
  "version": "1.0",
  "session_id": "0fb7d8cd-be55-431c-ac8c-026b6d6e03dd",
  "evaluated_at": "2025-11-21T10:30:00Z",
  "rubrics_version": "1.0",
  "rubric_scores": [
    {
      "rubric_id": "rubric_001",
      "rubric_name": "Task Completion Efficiency",
      "score": 4.0,
      "max_score": 5.0,
      "reasoning": "The user provided clear initial requirements and only needed minimal clarification. The task was completed in 3 turns with efficient tool usage."
    },
    {
      "rubric_id": "rubric_002",
      "rubric_name": "Clear Communication",
      "score": 5.0,
      "max_score": 5.0,
      "reasoning": "User instructions were exceptionally clear with specific file paths and expected behavior described upfront."
    }
  ],
  "summary": {
    "total_score": 4.5,
    "max_score": 5.0,
    "percentage": 90.0,
    "rubrics_evaluated": 2
  }
}
```

## Judge Template Prompt Example

```
prompts/judge_template.txt
--------------------------

You are an expert evaluator assessing AI agent interaction quality.

## Rubric: {rubric_name}

### Description
{rubric_description}

### Scoring Criteria
{scoring_criteria}

## Chat Session to Evaluate
{chat_session}

## Instructions
Evaluate the chat session above based ONLY on the rubric described.
Consider the user's efficiency in communicating with and directing the AI agent.

Provide your evaluation in the following format:
SCORE: [1-5]
REASONING: [2-3 sentences explaining your score]

Be objective and consistent. Focus on specific evidence from the chat session.
```

## Error Handling

1. **Invalid Rubric Format**: Validate rubrics on load, fail fast with clear message
2. **LLM Parse Failures**: If score cannot be parsed, retry once with explicit format reminder
3. **Timeout**: Set per-rubric timeout, mark as "evaluation_failed" if exceeded
4. **Rate Limits**: Queue with backoff, respect Gemini API limits

## Configuration Options

```python
@dataclass
class EvaluationConfig:
    max_concurrent: int = 10  # Max parallel LLM calls
    llm_temperature: float = 0.1  # Low temp for consistency
    llm_max_tokens: int = 1024
    timeout_per_rubric: int = 60  # seconds
    retry_on_parse_failure: bool = True
    score_scale: Tuple[int, int] = (1, 5)
```

## Batch Evaluation

For evaluating multiple sessions:

```python
# CLI command for batch evaluation
uv run retrochat-eval evaluate-batch \
    --rubrics ./rubrics.json \
    --sessions-dir ./sessions/ \
    --output-dir ./results/ \
    --parallel 5
```

Output structure:
```
results/
├── summary.json           # Aggregate statistics
├── session_001_result.json
├── session_002_result.json
└── session_003_result.json
```

## Metrics and Reporting

After batch evaluation, generate summary statistics:

```json
{
  "batch_summary": {
    "total_sessions": 50,
    "average_score": 3.8,
    "median_score": 4.0,
    "std_deviation": 0.7,
    "score_distribution": {
      "1": 2,
      "2": 5,
      "3": 12,
      "4": 20,
      "5": 11
    }
  },
  "per_rubric_summary": {
    "rubric_001": {
      "name": "Task Completion Efficiency",
      "average": 3.9,
      "median": 4.0
    },
    "rubric_002": {
      "name": "Clear Communication",
      "average": 4.2,
      "median": 4.0
    }
  }
}
```
