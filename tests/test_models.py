"""Tests for data models."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from retrochat_evaluator.models.rubric import Rubric, RubricList, TrainingConfig
from retrochat_evaluator.models.chat_session import ChatSession, Turn
from retrochat_evaluator.models.evaluation import (
    RubricScore,
    EvaluationResult,
    EvaluationSummary,
    BatchEvaluationSummary,
)


class TestRubric:
    """Tests for Rubric model."""

    def test_rubric_creation(self, sample_rubric: Rubric):
        """Test basic rubric creation."""
        assert sample_rubric.id == "rubric_001"
        assert sample_rubric.name == "Clear Communication"
        assert sample_rubric.weight == 1.0

    def test_rubric_default_weight(self):
        """Test that weight defaults to 1.0."""
        rubric = Rubric(
            id="test",
            name="Test",
            description="Test description",
            scoring_criteria="1-5 scale",
        )
        assert rubric.weight == 1.0

    def test_rubric_format_for_prompt(self, sample_rubric: Rubric):
        """Test rubric formatting for prompts."""
        formatted = sample_rubric.format_for_prompt()
        assert "Clear Communication" in formatted
        assert "clear and specific requirements" in formatted
        assert "Scoring Criteria:" in formatted


class TestRubricList:
    """Tests for RubricList model."""

    def test_rubric_list_creation(self, sample_rubric_list: RubricList):
        """Test RubricList creation."""
        assert len(sample_rubric_list.rubrics) == 3
        assert sample_rubric_list.version == "1.0"

    def test_get_rubric(self, sample_rubric_list: RubricList):
        """Test getting rubric by ID."""
        rubric = sample_rubric_list.get_rubric("rubric_001")
        assert rubric is not None
        assert rubric.name == "Clear Communication"

    def test_get_rubric_not_found(self, sample_rubric_list: RubricList):
        """Test getting non-existent rubric."""
        rubric = sample_rubric_list.get_rubric("nonexistent")
        assert rubric is None

    def test_rubric_list_json_roundtrip(self, sample_rubric_list: RubricList):
        """Test saving and loading RubricList to/from JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_rubric_list.to_json(temp_path)
            loaded = RubricList.from_json(temp_path)

            assert len(loaded.rubrics) == len(sample_rubric_list.rubrics)
            assert loaded.version == sample_rubric_list.version
            assert loaded.rubrics[0].name == sample_rubric_list.rubrics[0].name
        finally:
            temp_path.unlink()

    def test_format_all_for_prompt(self, sample_rubric_list: RubricList):
        """Test formatting all rubrics for prompt."""
        formatted = sample_rubric_list.format_all_for_prompt()
        assert "Rubric 1" in formatted
        assert "Rubric 2" in formatted
        assert "Clear Communication" in formatted
        assert "Task Efficiency" in formatted


class TestChatSession:
    """Tests for ChatSession model."""

    def test_chat_session_creation(self, sample_chat_session: ChatSession):
        """Test ChatSession creation."""
        assert sample_chat_session.session_id == "test-session-001"
        assert len(sample_chat_session.turns) == 8
        assert sample_chat_session.total_turns == 2

    def test_chat_session_turn_count(self, sample_chat_session: ChatSession):
        """Test turn count calculation."""
        # max turn_number = 2
        assert sample_chat_session.turn_count == 2

    def test_chat_session_from_jsonl(self, mock_session_path: Path):
        """Test parsing ChatSession from JSONL."""
        session = ChatSession.from_jsonl(mock_session_path)

        assert session.session_id == "test-session-001"
        assert len(session.turns) > 0

    def test_chat_session_from_json(self, tmp_path: Path):
        """Test parsing ChatSession from JSON."""
        json_data = {
            "session_id": "test-json-001",
            "total_turns": 1,
            "turns": [
                {
                    "turn_number": 1,
                    "role": "user",
                    "message_type": "simple_message",
                    "content": "Hello world",
                }
            ],
        }
        json_path = tmp_path / "test_session.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        session = ChatSession.from_json(json_path)
        assert session.session_id == "test-json-001"
        assert session.total_turns == 1
        assert len(session.turns) == 1
        assert session.turns[0].content == "Hello world"

    def test_chat_session_format_for_prompt(self, sample_chat_session: ChatSession):
        """Test formatting session for prompt (now returns JSON)."""
        formatted = sample_chat_session.format_for_prompt()

        # format_for_prompt now returns JSON
        parsed = json.loads(formatted)
        assert parsed["session_id"] == "test-session-001"
        assert "turns" in parsed
        assert len(parsed["turns"]) == 8


class TestTurn:
    """Tests for Turn model."""

    def test_turn_creation(self):
        """Test Turn creation."""
        turn = Turn(
            turn_number=1,
            role="user",
            message_type="simple_message",
            content="Hello world",
        )
        assert turn.role == "user"
        assert turn.message_type == "simple_message"

    def test_turn_tool_request(self):
        """Test Turn with tool_request message type."""
        turn = Turn(
            turn_number=1,
            role="assistant",
            message_type="tool_request(Read)",
            content="file_path: /test/file.py",
        )
        assert turn.message_type == "tool_request(Read)"
        assert "/test/file.py" in turn.content

    def test_turn_tool_result(self):
        """Test Turn with tool_result message type."""
        turn = Turn(
            turn_number=1,
            role="user",
            message_type="tool_result(Read)",
            content="file contents here",
        )
        assert turn.message_type == "tool_result(Read)"
        assert turn.role == "user"


class TestRubricScore:
    """Tests for RubricScore model."""

    def test_rubric_score_creation(self):
        """Test RubricScore creation."""
        score = RubricScore(
            rubric_id="rubric_001",
            rubric_name="Test",
            score=4.0,
            max_score=5.0,
            reasoning="Good performance.",
        )
        assert score.score == 4.0
        assert score.percentage == 80.0

    def test_rubric_score_validation(self):
        """Test score validation (1-5 range)."""
        # Valid scores
        RubricScore(rubric_id="r1", rubric_name="Test", score=1.0, reasoning="Min")
        RubricScore(rubric_id="r1", rubric_name="Test", score=5.0, reasoning="Max")

        # Invalid scores should raise
        with pytest.raises(ValueError):
            RubricScore(rubric_id="r1", rubric_name="Test", score=0.0, reasoning="Too low")
        with pytest.raises(ValueError):
            RubricScore(rubric_id="r1", rubric_name="Test", score=6.0, reasoning="Too high")


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_evaluation_result_creation(self, sample_rubric_scores: list[RubricScore]):
        """Test EvaluationResult creation."""
        result = EvaluationResult(
            session_id="test-001",
            rubric_scores=sample_rubric_scores,
        )
        assert result.session_id == "test-001"
        assert len(result.rubric_scores) == 3

    def test_calculate_summary(
        self, sample_rubric_scores: list[RubricScore], sample_rubrics: list[Rubric]
    ):
        """Test summary calculation."""
        result = EvaluationResult(
            session_id="test-001",
            rubric_scores=sample_rubric_scores,
        )

        weights = {r.id: r.weight for r in sample_rubrics}
        summary = result.calculate_summary(weights=weights)

        assert summary.rubrics_evaluated == 3
        assert 3.0 <= summary.total_score <= 5.0
        assert summary.max_score == 5.0

    def test_evaluation_result_json_roundtrip(self, sample_rubric_scores: list[RubricScore]):
        """Test saving and loading EvaluationResult."""
        result = EvaluationResult(
            session_id="test-001",
            rubric_scores=sample_rubric_scores,
        )
        result.calculate_summary()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result.to_json(temp_path)
            loaded = EvaluationResult.from_json(temp_path)

            assert loaded.session_id == result.session_id
            assert len(loaded.rubric_scores) == len(result.rubric_scores)
        finally:
            temp_path.unlink()


class TestBatchEvaluationSummary:
    """Tests for BatchEvaluationSummary model."""

    def test_from_results(self, sample_rubric_scores: list[RubricScore]):
        """Test creating batch summary from results."""
        results = [
            EvaluationResult(session_id=f"session-{i}", rubric_scores=sample_rubric_scores)
            for i in range(5)
        ]
        for r in results:
            r.calculate_summary()

        summary = BatchEvaluationSummary.from_results(results)

        assert summary.total_sessions == 5
        assert summary.average_score > 0
        assert summary.median_score > 0

    def test_empty_results(self):
        """Test batch summary with no results."""
        summary = BatchEvaluationSummary.from_results([])
        assert summary.total_sessions == 0
        assert summary.average_score == 0
