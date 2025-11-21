"""Tests for validation module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrochat_evaluator.models.validation import (
    SessionValidationResult,
    ValidationMetrics,
    ValidationReport,
)
from retrochat_evaluator.models.rubric import Rubric, RubricList
from retrochat_evaluator.models.evaluation import EvaluationResult, EvaluationSummary, RubricScore
from retrochat_evaluator.validation.validator import Validator


class TestSessionValidationResult:
    """Tests for SessionValidationResult model."""

    def test_creation(self):
        """Test creating a session validation result."""
        result = SessionValidationResult(
            session_id="test-001",
            file="session.jsonl",
            predicted_score=4.2,
            real_score=4.5,
            error=-0.3,
            absolute_error=0.3,
            squared_error=0.09,
        )

        assert result.session_id == "test-001"
        assert result.file == "session.jsonl"
        assert result.predicted_score == 4.2
        assert result.real_score == 4.5
        assert result.error == -0.3
        assert result.absolute_error == 0.3
        assert result.squared_error == 0.09


class TestValidationMetrics:
    """Tests for ValidationMetrics model."""

    def test_creation(self):
        """Test creating validation metrics."""
        metrics = ValidationMetrics(
            mean_absolute_error=0.45,
            root_mean_squared_error=0.58,
            mean_error=-0.12,
            std_error=0.35,
            correlation=0.92,
            r_squared=0.85,
            min_error=-0.8,
            max_error=0.5,
        )

        assert metrics.mean_absolute_error == 0.45
        assert metrics.root_mean_squared_error == 0.58
        assert metrics.mean_error == -0.12
        assert metrics.std_error == 0.35
        assert metrics.correlation == 0.92
        assert metrics.r_squared == 0.85
        assert metrics.min_error == -0.8
        assert metrics.max_error == 0.5

    def test_creation_minimal(self):
        """Test creating metrics without optional fields."""
        metrics = ValidationMetrics(
            mean_absolute_error=0.5,
            root_mean_squared_error=0.6,
            mean_error=0.1,
            min_error=-0.5,
            max_error=0.7,
        )

        assert metrics.mean_absolute_error == 0.5
        assert metrics.std_error is None
        assert metrics.correlation is None
        assert metrics.r_squared is None


class TestValidationReport:
    """Tests for ValidationReport model."""

    @pytest.fixture
    def sample_session_results(self) -> list[SessionValidationResult]:
        """Create sample session validation results."""
        return [
            SessionValidationResult(
                session_id="session-001",
                file="session1.jsonl",
                predicted_score=4.2,
                real_score=4.5,
                error=-0.3,
                absolute_error=0.3,
                squared_error=0.09,
            ),
            SessionValidationResult(
                session_id="session-002",
                file="session2.jsonl",
                predicted_score=3.8,
                real_score=3.5,
                error=0.3,
                absolute_error=0.3,
                squared_error=0.09,
            ),
            SessionValidationResult(
                session_id="session-003",
                file="session3.jsonl",
                predicted_score=4.0,
                real_score=4.0,
                error=0.0,
                absolute_error=0.0,
                squared_error=0.0,
            ),
        ]

    def test_from_results(self, sample_session_results: list[SessionValidationResult]):
        """Test creating report from results."""
        report = ValidationReport.from_results(
            session_results=sample_session_results,
            score_name="efficiency",
            rubrics_file="./rubrics.json",
        )

        assert report.total_sessions == 3
        assert report.score_name == "efficiency"
        assert report.rubrics_file == "./rubrics.json"
        assert len(report.session_results) == 3

        # Check metrics
        # MAE = (0.3 + 0.3 + 0.0) / 3 = 0.2
        assert report.metrics.mean_absolute_error == 0.2

        # RMSE = sqrt((0.09 + 0.09 + 0.0) / 3) = sqrt(0.06) ≈ 0.2449
        assert 0.24 <= report.metrics.root_mean_squared_error <= 0.25

        # Mean error = (-0.3 + 0.3 + 0.0) / 3 = 0.0
        assert report.metrics.mean_error == 0.0

        # Error range
        assert report.metrics.min_error == -0.3
        assert report.metrics.max_error == 0.3

    def test_from_results_empty(self):
        """Test creating report from empty results."""
        report = ValidationReport.from_results(
            session_results=[],
            score_name="efficiency",
            rubrics_file="./rubrics.json",
        )

        assert report.total_sessions == 0
        assert report.metrics.mean_absolute_error == 0
        assert report.metrics.root_mean_squared_error == 0
        assert report.session_results == []

    def test_from_results_single_session(self):
        """Test creating report with single session (no std_error)."""
        result = SessionValidationResult(
            session_id="session-001",
            file="session1.jsonl",
            predicted_score=4.2,
            real_score=4.5,
            error=-0.3,
            absolute_error=0.3,
            squared_error=0.09,
        )

        report = ValidationReport.from_results(
            session_results=[result],
            score_name="efficiency",
            rubrics_file="./rubrics.json",
        )

        assert report.total_sessions == 1
        assert report.metrics.std_error is None  # Can't calculate with single value
        assert report.metrics.correlation is None  # Can't calculate with single value

    def test_calculate_correlation(self):
        """Test correlation calculation."""
        # Perfect positive correlation
        results = [
            SessionValidationResult(
                session_id=f"s{i}",
                file=f"s{i}.jsonl",
                predicted_score=float(i),
                real_score=float(i),
                error=0.0,
                absolute_error=0.0,
                squared_error=0.0,
            )
            for i in range(1, 6)
        ]

        report = ValidationReport.from_results(
            session_results=results,
            score_name="test",
            rubrics_file="test.json",
        )

        assert report.metrics.correlation == 1.0
        assert report.metrics.r_squared == 1.0

    def test_json_roundtrip(self, sample_session_results: list[SessionValidationResult]):
        """Test JSON serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            report = ValidationReport.from_results(
                session_results=sample_session_results,
                score_name="efficiency",
                rubrics_file="./rubrics.json",
            )

            report.to_json(output_path)
            assert output_path.exists()

            loaded_report = ValidationReport.from_json(output_path)
            assert loaded_report.total_sessions == report.total_sessions
            assert loaded_report.score_name == report.score_name
            assert len(loaded_report.session_results) == len(report.session_results)
            assert loaded_report.metrics.mean_absolute_error == report.metrics.mean_absolute_error


class TestValidator:
    """Tests for Validator orchestrator."""

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

    @pytest.fixture
    def temp_rubrics_file(self, sample_rubric_list: RubricList) -> Path:
        """Create temporary rubrics file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            sample_rubric_list.to_json(Path(f.name))
            return Path(f.name)

    @pytest.mark.asyncio
    async def test_validate(
        self,
        prompts_dir: Path,
        fixtures_dir: Path,
        temp_rubrics_file: Path,
        mock_judge_response: str,
    ):
        """Test full validation pipeline."""
        manifest_path = fixtures_dir / "mock_manifest.json"

        with patch("retrochat_evaluator.evaluation.evaluator.GeminiClient") as MockClient:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_judge_response)
            MockClient.return_value = mock_client

            validator = Validator(
                dataset_dir=fixtures_dir,
                manifest_path=manifest_path,
                rubrics_path=temp_rubrics_file,
                prompts_dir=prompts_dir,
                score_name="efficiency",
            )

            report = await validator.validate(max_sessions=2)

            assert isinstance(report, ValidationReport)
            assert report.total_sessions == 2
            assert report.score_name == "efficiency"
            assert len(report.session_results) == 2
            assert report.metrics is not None

    @pytest.mark.asyncio
    async def test_validate_no_matching_score(
        self,
        prompts_dir: Path,
        fixtures_dir: Path,
        temp_rubrics_file: Path,
    ):
        """Test validation with non-existent score name."""
        manifest_path = fixtures_dir / "mock_manifest.json"

        validator = Validator(
            dataset_dir=fixtures_dir,
            manifest_path=manifest_path,
            rubrics_path=temp_rubrics_file,
            prompts_dir=prompts_dir,
            score_name="nonexistent_score",
        )

        report = await validator.validate()

        assert report.total_sessions == 0
        assert len(report.session_results) == 0

    @pytest.mark.asyncio
    async def test_validate_all_sessions(
        self,
        prompts_dir: Path,
        fixtures_dir: Path,
        temp_rubrics_file: Path,
        mock_judge_response: str,
    ):
        """Test validating all sessions without max_sessions limit."""
        manifest_path = fixtures_dir / "mock_manifest.json"

        with patch("retrochat_evaluator.evaluation.evaluator.GeminiClient") as MockClient:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_judge_response)
            MockClient.return_value = mock_client

            validator = Validator(
                dataset_dir=fixtures_dir,
                manifest_path=manifest_path,
                rubrics_path=temp_rubrics_file,
                prompts_dir=prompts_dir,
                score_name="efficiency",
            )

            # Note: mock_manifest.json has 3 sessions, but low_score_session.jsonl might not exist
            # The validator should handle missing files gracefully
            report = await validator.validate()

            assert isinstance(report, ValidationReport)
            assert report.score_name == "efficiency"

    def test_save_report(
        self,
        prompts_dir: Path,
        fixtures_dir: Path,
        temp_rubrics_file: Path,
    ):
        """Test saving validation report."""
        manifest_path = fixtures_dir / "mock_manifest.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            validator = Validator(
                dataset_dir=fixtures_dir,
                manifest_path=manifest_path,
                rubrics_path=temp_rubrics_file,
                prompts_dir=prompts_dir,
                score_name="efficiency",
            )

            report = ValidationReport.from_results(
                session_results=[
                    SessionValidationResult(
                        session_id="test",
                        file="test.jsonl",
                        predicted_score=4.0,
                        real_score=4.5,
                        error=-0.5,
                        absolute_error=0.5,
                        squared_error=0.25,
                    )
                ],
                score_name="efficiency",
                rubrics_file=str(temp_rubrics_file),
            )

            validator.save_report(report, output_path)
            assert output_path.exists()

            loaded = ValidationReport.from_json(output_path)
            assert loaded.total_sessions == 1


class TestValidationReportCorrelation:
    """Additional tests for correlation calculation edge cases."""

    def test_negative_correlation(self):
        """Test negative correlation detection."""
        # Create results with negative correlation
        results = [
            SessionValidationResult(
                session_id="s1",
                file="s1.jsonl",
                predicted_score=5.0,
                real_score=1.0,
                error=4.0,
                absolute_error=4.0,
                squared_error=16.0,
            ),
            SessionValidationResult(
                session_id="s2",
                file="s2.jsonl",
                predicted_score=1.0,
                real_score=5.0,
                error=-4.0,
                absolute_error=4.0,
                squared_error=16.0,
            ),
        ]

        report = ValidationReport.from_results(
            session_results=results,
            score_name="test",
            rubrics_file="test.json",
        )

        assert report.metrics.correlation is not None
        assert report.metrics.correlation < 0  # Should be negative

    def test_zero_variance(self):
        """Test when all predicted scores are the same (zero variance)."""
        results = [
            SessionValidationResult(
                session_id=f"s{i}",
                file=f"s{i}.jsonl",
                predicted_score=3.0,  # All same
                real_score=float(i),
                error=3.0 - float(i),
                absolute_error=abs(3.0 - float(i)),
                squared_error=(3.0 - float(i)) ** 2,
            )
            for i in range(1, 4)
        ]

        report = ValidationReport.from_results(
            session_results=results,
            score_name="test",
            rubrics_file="test.json",
        )

        # Correlation should be None when variance is zero
        assert report.metrics.correlation is None
        assert report.metrics.r_squared is None
