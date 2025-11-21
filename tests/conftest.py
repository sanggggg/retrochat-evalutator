"""Shared pytest fixtures for retrochat-evaluator tests."""

import json
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from retrochat_evaluator.models.rubric import Rubric, RubricList
from retrochat_evaluator.models.chat_session import ChatSession, ChatMessage, ToolCall
from retrochat_evaluator.models.evaluation import RubricScore, EvaluationResult
from retrochat_evaluator.llm.gemini import GeminiClient
from retrochat_evaluator.config import LLMConfig


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def mock_session_path(fixtures_dir: Path) -> Path:
    """Return path to mock session JSONL file."""
    return fixtures_dir / "mock_session.jsonl"


@pytest.fixture
def mock_session_2_path(fixtures_dir: Path) -> Path:
    """Return path to second mock session JSONL file."""
    return fixtures_dir / "mock_session_2.jsonl"


@pytest.fixture
def mock_manifest_path(fixtures_dir: Path) -> Path:
    """Return path to mock manifest JSON file."""
    return fixtures_dir / "mock_manifest.json"


@pytest.fixture
def sample_rubric() -> Rubric:
    """Create a sample rubric for testing."""
    return Rubric(
        id="rubric_001",
        name="Clear Communication",
        description="The user provides clear and specific requirements",
        scoring_criteria="1: Vague\n2: Somewhat clear\n3: Clear\n4: Very clear\n5: Excellent",
        weight=1.0,
    )


@pytest.fixture
def sample_rubrics() -> list[Rubric]:
    """Create a list of sample rubrics for testing."""
    return [
        Rubric(
            id="rubric_001",
            name="Clear Communication",
            description="The user provides clear and specific requirements",
            scoring_criteria="1: Vague\n2: Somewhat clear\n3: Clear\n4: Very clear\n5: Excellent",
            weight=1.0,
        ),
        Rubric(
            id="rubric_002",
            name="Task Efficiency",
            description="The user guides the AI efficiently toward the solution",
            scoring_criteria="1: Inefficient\n2: Somewhat efficient\n3: Average\n4: Efficient\n5: Optimal",
            weight=1.5,
        ),
        Rubric(
            id="rubric_003",
            name="Context Provision",
            description="The user provides necessary context and information",
            scoring_criteria="1: No context\n2: Minimal\n3: Adequate\n4: Good\n5: Comprehensive",
            weight=1.0,
        ),
    ]


@pytest.fixture
def sample_rubric_list(sample_rubrics: list[Rubric]) -> RubricList:
    """Create a sample RubricList for testing."""
    return RubricList(
        version="1.0",
        created_at=datetime(2025, 11, 21, 10, 0, 0),
        rubrics=sample_rubrics,
    )


@pytest.fixture
def sample_chat_session() -> ChatSession:
    """Create a sample ChatSession for testing."""
    return ChatSession(
        session_id="test-session-001",
        messages=[
            ChatMessage(
                role="user",
                content="Please add a function to calculate the sum of two numbers",
                timestamp=datetime(2025, 11, 6, 10, 0, 0),
            ),
            ChatMessage(
                role="assistant",
                content="I'll help you create that function.",
                tool_calls=[
                    ToolCall(
                        tool_id="toolu_001",
                        tool_name="Read",
                        tool_input={"file_path": "/test/math.py"},
                        tool_result="def multiply(a, b):\n    return a * b",
                    )
                ],
                timestamp=datetime(2025, 11, 6, 10, 0, 5),
            ),
            ChatMessage(
                role="assistant",
                content="Done! I've added the calculate_sum function.",
                tool_calls=[
                    ToolCall(
                        tool_id="toolu_002",
                        tool_name="Edit",
                        tool_input={"file_path": "/test/math.py", "old_string": "...", "new_string": "..."},
                        tool_result="File edited successfully",
                    )
                ],
                timestamp=datetime(2025, 11, 6, 10, 0, 10),
            ),
            ChatMessage(
                role="user",
                content="Perfect, thanks!",
                timestamp=datetime(2025, 11, 6, 10, 0, 15),
            ),
        ],
        cwd="/Users/test/project",
        git_branch="main",
        start_timestamp=datetime(2025, 11, 6, 10, 0, 0),
    )


@pytest.fixture
def sample_rubric_scores(sample_rubrics: list[Rubric]) -> list[RubricScore]:
    """Create sample rubric scores for testing."""
    return [
        RubricScore(
            rubric_id="rubric_001",
            rubric_name="Clear Communication",
            score=4.0,
            max_score=5.0,
            reasoning="The user provided clear initial requirements.",
        ),
        RubricScore(
            rubric_id="rubric_002",
            rubric_name="Task Efficiency",
            score=5.0,
            max_score=5.0,
            reasoning="The task was completed efficiently in minimal turns.",
        ),
        RubricScore(
            rubric_id="rubric_003",
            rubric_name="Context Provision",
            score=3.0,
            max_score=5.0,
            reasoning="Adequate context was provided.",
        ),
    ]


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock()
    client.generate_batch = AsyncMock()
    return client


@pytest.fixture
def mock_extractor_response() -> str:
    """Sample LLM response for rubric extraction."""
    return '''Based on my analysis of this chat session, here are the evaluation rubrics:

```json
[
  {
    "name": "Clear Requirements",
    "description": "The user clearly states what they want the AI to accomplish",
    "scoring_criteria": "1: Vague, 2: Somewhat clear, 3: Clear, 4: Very clear, 5: Excellent",
    "evidence": "The user asked for a specific function with defined behavior"
  },
  {
    "name": "Efficient Interaction",
    "description": "The user guides the AI efficiently without unnecessary back-and-forth",
    "scoring_criteria": "1: Many unnecessary exchanges, 2: Some inefficiency, 3: Average, 4: Efficient, 5: Optimal",
    "evidence": "Task completed in 2 turns"
  }
]
```
'''


@pytest.fixture
def mock_summarizer_response() -> str:
    """Sample LLM response for rubric summarization."""
    return '''Here is the consolidated set of rubrics:

```json
{
  "rubrics": [
    {
      "id": "rubric_001",
      "name": "Clear Initial Requirements",
      "description": "The user provides complete, unambiguous requirements in their initial request",
      "scoring_criteria": "1: Vague request, 2: Partial requirements, 3: Adequate requirements, 4: Clear requirements, 5: Comprehensive requirements",
      "weight": 1.0
    },
    {
      "id": "rubric_002",
      "name": "Task Efficiency",
      "description": "The user guides the AI efficiently toward the solution",
      "scoring_criteria": "1: Very inefficient, 2: Somewhat inefficient, 3: Average, 4: Efficient, 5: Optimal",
      "weight": 1.5
    }
  ],
  "consolidation_notes": "Merged similar rubrics about clarity and efficiency"
}
```
'''


@pytest.fixture
def mock_judge_response() -> str:
    """Sample LLM response for judge scoring."""
    return """SCORE: 4
REASONING: The user provided clear initial requirements specifying the function name and expected behavior. The task was completed efficiently with minimal back-and-forth."""
