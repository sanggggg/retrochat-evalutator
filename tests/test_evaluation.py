"""Tests for evaluation module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrochat_evaluator.evaluation.judge import JudgePromptGenerator
from retrochat_evaluator.evaluation.scorer import RubricScorer
from retrochat_evaluator.evaluation.aggregator import ResultAggregator
from retrochat_evaluator.evaluation.evaluator import Evaluator
from retrochat_evaluator.models.rubric import Rubric
from retrochat_evaluator.models.chat_session import ChatSession
from retrochat_evaluator.models.evaluation import RubricScore, EvaluationResult
from retrochat_evaluator.config import EvaluationConfig


class TestJudgePromptGenerator:
    """Tests for JudgePromptGenerator."""

    @pytest.fixture
    def judge_template_path(self) -> Path:
        """Create temporary judge template."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Rubric: {rubric_name}\n"
                "Description: {rubric_description}\n"
                "Criteria: {scoring_criteria}\n"
                "Session:\n{chat_session}\n"
                "Respond with SCORE and REASONING."
            )
            return Path(f.name)

    def test_generate(
        self,
        judge_template_path: Path,
        sample_rubric: Rubric,
        sample_chat_session: ChatSession,
    ):
        """Test generating judge prompt."""
        generator = JudgePromptGenerator(template_path=judge_template_path)
        prompt = generator.generate(sample_rubric, sample_chat_session)

        assert "Clear Communication" in prompt
        assert "clear and specific requirements" in prompt
        assert "test-session-001" in prompt

    def test_generate_all(
        self,
        judge_template_path: Path,
        sample_rubrics: list[Rubric],
        sample_chat_session: ChatSession,
    ):
        """Test generating prompts for all rubrics."""
        generator = JudgePromptGenerator(template_path=judge_template_path)
        rubric_prompts = generator.generate_all(sample_rubrics, sample_chat_session)

        assert len(rubric_prompts) == 3
        assert all(isinstance(rp, tuple) and len(rp) == 2 for rp in rubric_prompts)
        assert all(isinstance(rp[0], Rubric) and isinstance(rp[1], str) for rp in rubric_prompts)


class TestRubricScorer:
    """Tests for RubricScorer."""

    @pytest.mark.asyncio
    async def test_score(
        self,
        mock_llm_client: MagicMock,
        sample_rubric: Rubric,
        mock_judge_response: str,
    ):
        """Test scoring a single rubric."""
        mock_llm_client.generate.return_value = mock_judge_response

        scorer = RubricScorer(llm_client=mock_llm_client)
        score = await scorer.score(sample_rubric, "test prompt")

        assert isinstance(score, RubricScore)
        assert score.rubric_id == "rubric_001"
        assert score.score == 4.0
        assert "clear initial requirements" in score.reasoning.lower()

    @pytest.mark.asyncio
    async def test_score_all(
        self,
        mock_llm_client: MagicMock,
        sample_rubrics: list[Rubric],
        mock_judge_response: str,
    ):
        """Test scoring all rubrics in parallel."""
        mock_llm_client.generate.return_value = mock_judge_response

        scorer = RubricScorer(llm_client=mock_llm_client)
        rubric_prompts = [(r, f"prompt for {r.id}") for r in sample_rubrics]
        scores = await scorer.score_all(rubric_prompts)

        assert len(scores) == 3
        assert all(isinstance(s, RubricScore) for s in scores)

    @pytest.mark.asyncio
    async def test_score_parse_failure_retry(
        self,
        mock_llm_client: MagicMock,
        sample_rubric: Rubric,
        mock_judge_response: str,
    ):
        """Test retry on parse failure."""
        # First call returns unparseable, second returns valid
        mock_llm_client.generate.side_effect = [
            "Invalid response without score",
            mock_judge_response,
        ]

        config = EvaluationConfig(retry_on_parse_failure=True)
        scorer = RubricScorer(llm_client=mock_llm_client, config=config)
        score = await scorer.score(sample_rubric, "test prompt")

        assert score.score == 4.0
        assert mock_llm_client.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_score_default_on_failure(
        self,
        mock_llm_client: MagicMock,
        sample_rubric: Rubric,
    ):
        """Test default score when parsing fails."""
        mock_llm_client.generate.return_value = "Completely unparseable response"

        config = EvaluationConfig(retry_on_parse_failure=False)
        scorer = RubricScorer(llm_client=mock_llm_client, config=config)
        score = await scorer.score(sample_rubric, "test prompt")

        assert score.score == 3.0  # Default middle score
        assert "Unable to parse" in score.reasoning

    def test_parse_score_valid(self, mock_llm_client: MagicMock):
        """Test parsing valid score response."""
        scorer = RubricScorer(llm_client=mock_llm_client)

        response = "SCORE: 5\nREASONING: Excellent work throughout."
        score, reasoning = scorer._parse_score(response)

        assert score == 5.0
        assert "Excellent work" in reasoning

    def test_parse_score_with_decimal(self, mock_llm_client: MagicMock):
        """Test parsing score with decimal."""
        scorer = RubricScorer(llm_client=mock_llm_client)

        response = "SCORE: 4.5\nREASONING: Very good."
        score, reasoning = scorer._parse_score(response)

        assert score == 4.5

    def test_parse_score_clamps_to_range(self, mock_llm_client: MagicMock):
        """Test score clamping to valid range."""
        scorer = RubricScorer(llm_client=mock_llm_client)

        # Too high
        response = "SCORE: 10\nREASONING: Perfect."
        score, _ = scorer._parse_score(response)
        assert score == 5.0

        # Too low
        response = "SCORE: 0\nREASONING: Terrible."
        score, _ = scorer._parse_score(response)
        assert score == 1.0

    def test_parse_score_invalid(self, mock_llm_client: MagicMock):
        """Test parsing invalid response."""
        scorer = RubricScorer(llm_client=mock_llm_client)

        response = "This is not a valid score response."
        score, reasoning = scorer._parse_score(response)

        assert score is None
        assert reasoning == ""


class TestResultAggregator:
    """Tests for ResultAggregator."""

    def test_aggregate(
        self,
        sample_rubric_scores: list[RubricScore],
        sample_rubrics: list[Rubric],
    ):
        """Test aggregating scores."""
        aggregator = ResultAggregator()
        result = aggregator.aggregate(
            session_id="test-001",
            rubric_scores=sample_rubric_scores,
            rubrics=sample_rubrics,
            rubrics_version="1.0",
        )

        assert isinstance(result, EvaluationResult)
        assert result.session_id == "test-001"
        assert result.rubrics_version == "1.0"
        assert result.summary is not None
        assert result.summary.rubrics_evaluated == 3

    def test_aggregate_with_weights(
        self,
        sample_rubric_scores: list[RubricScore],
        sample_rubrics: list[Rubric],
    ):
        """Test aggregation considers weights."""
        aggregator = ResultAggregator()
        result = aggregator.aggregate(
            session_id="test-001",
            rubric_scores=sample_rubric_scores,
            rubrics=sample_rubrics,
        )

        # Rubric 002 has weight 1.5, others have 1.0
        # Scores: 4.0 (w=1.0), 5.0 (w=1.5), 3.0 (w=1.0)
        # Weighted avg: (4.0*1.0 + 5.0*1.5 + 3.0*1.0) / (1.0 + 1.5 + 1.0)
        #             = (4.0 + 7.5 + 3.0) / 3.5 = 14.5 / 3.5 ≈ 4.14
        assert 4.0 <= result.summary.total_score <= 4.3

    def test_aggregate_batch(
        self,
        sample_rubric_scores: list[RubricScore],
        sample_rubrics: list[Rubric],
    ):
        """Test batch aggregation."""
        aggregator = ResultAggregator()
        results_data = [
            ("session-1", sample_rubric_scores),
            ("session-2", sample_rubric_scores),
        ]
        results = aggregator.aggregate_batch(results_data, sample_rubrics)

        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)


class TestEvaluator:
    """Tests for Evaluator orchestrator."""

    @pytest.fixture
    def prompts_dir(self) -> Path:
        """Create temporary prompts directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir)
            (prompts_dir / "judge_template.txt").write_text(
                "Rubric: {rubric_name}\n"
                "Description: {rubric_description}\n"
                "Criteria: {scoring_criteria}\n"
                "Session:\n{chat_session}\n"
            )
            yield prompts_dir

    @pytest.mark.asyncio
    async def test_evaluate(
        self,
        prompts_dir: Path,
        sample_rubrics: list[Rubric],
        sample_chat_session: ChatSession,
        mock_judge_response: str,
    ):
        """Test full evaluation pipeline."""
        with patch("retrochat_evaluator.evaluation.evaluator.GeminiClient") as MockClient:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_judge_response)
            MockClient.return_value = mock_client

            evaluator = Evaluator(prompts_dir=prompts_dir)
            result = await evaluator.evaluate(
                rubrics=sample_rubrics,
                session=sample_chat_session,
                rubrics_version="1.0",
            )

            assert isinstance(result, EvaluationResult)
            assert result.session_id == "test-session-001"
            assert len(result.rubric_scores) == 3
            assert result.summary is not None

    @pytest.mark.asyncio
    async def test_evaluate_batch(
        self,
        prompts_dir: Path,
        sample_rubrics: list[Rubric],
        sample_chat_session: ChatSession,
        mock_judge_response: str,
    ):
        """Test batch evaluation."""
        with patch("retrochat_evaluator.evaluation.evaluator.GeminiClient") as MockClient:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_judge_response)
            MockClient.return_value = mock_client

            evaluator = Evaluator(prompts_dir=prompts_dir)
            sessions = [sample_chat_session, sample_chat_session]
            results = await evaluator.evaluate_batch(
                rubrics=sample_rubrics,
                sessions=sessions,
            )

            assert len(results) == 2
            assert all(isinstance(r, EvaluationResult) for r in results)

    def test_save_result(
        self,
        prompts_dir: Path,
        sample_rubric_scores: list[RubricScore],
    ):
        """Test saving evaluation result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.json"

            evaluator = Evaluator(prompts_dir=prompts_dir)
            result = EvaluationResult(
                session_id="test-001",
                rubric_scores=sample_rubric_scores,
            )
            result.calculate_summary()

            evaluator.save_result(result, output_path)
            assert output_path.exists()

    def test_save_batch_results(
        self,
        prompts_dir: Path,
        sample_rubric_scores: list[RubricScore],
    ):
        """Test saving batch results with summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"

            evaluator = Evaluator(prompts_dir=prompts_dir)
            results = []
            for i in range(3):
                r = EvaluationResult(
                    session_id=f"session-{i}",
                    rubric_scores=sample_rubric_scores,
                )
                r.calculate_summary()
                results.append(r)

            summary_path = evaluator.save_batch_results(results, output_dir)

            assert output_dir.exists()
            assert summary_path.exists()
            assert (output_dir / "session-0_result.json").exists()
            assert (output_dir / "session-1_result.json").exists()
            assert (output_dir / "session-2_result.json").exists()
