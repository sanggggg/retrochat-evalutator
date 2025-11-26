"""Tests for CLI commands."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from retrochat_evaluator.cli import main
from retrochat_evaluator.models.validation import (
    SessionValidationResult,
    ValidationMetrics,
    ValidationReport,
)


class TestValidationMetricsCommand:
    """Tests for validation-metrics CLI command."""

    @pytest.fixture
    def sample_validation_report(self) -> ValidationReport:
        """Create a sample validation report."""
        session_results = [
            SessionValidationResult(
                session_id="session-001",
                file="session1.jsonl",
                predicted_score=4.2,
                real_score=4.5,
            ),
            SessionValidationResult(
                session_id="session-002",
                file="session2.jsonl",
                predicted_score=3.8,
                real_score=3.5,
            ),
            SessionValidationResult(
                session_id="session-003",
                file="session3.jsonl",
                predicted_score=4.0,
                real_score=4.0,
            ),
        ]

        return ValidationReport.from_results(
            session_results=session_results,
            score_name="efficiency",
            rubrics_file="./rubrics.json",
        )

    def test_validation_metrics_basic(self, sample_validation_report: ValidationReport):
        """Test basic validation-metrics command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            sample_validation_report.to_json(report_path)

            runner = CliRunner()
            result = runner.invoke(main, ["validation-metrics", str(report_path)])

            assert result.exit_code == 0
            assert "Validation Report:" in result.output
            assert "Total sessions: 3" in result.output
            assert "Score type: efficiency" in result.output
            assert "Kendall Rank Correlation" in result.output
            assert "P-value:" in result.output

    def test_validation_metrics_with_show_sessions(
        self, sample_validation_report: ValidationReport
    ):
        """Test validation-metrics command with --show-sessions flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            sample_validation_report.to_json(report_path)

            runner = CliRunner()
            result = runner.invoke(main, ["validation-metrics", str(report_path), "--show-sessions"])

            assert result.exit_code == 0
            assert "Per-Session Results:" in result.output
            assert "session1.jsonl" in result.output
            assert "session2.jsonl" in result.output
            assert "session3.jsonl" in result.output
            assert "4.2000" in result.output  # predicted_score
            assert "4.5000" in result.output  # real_score

    def test_validation_metrics_file_not_found(self):
        """Test validation-metrics command with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(main, ["validation-metrics", "nonexistent.json"])

        assert result.exit_code != 0

    def test_validation_metrics_empty_report(self):
        """Test validation-metrics command with empty report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            empty_report = ValidationReport.from_results(
                session_results=[],
                score_name="efficiency",
                rubrics_file="./rubrics.json",
            )
            empty_report.to_json(report_path)

            runner = CliRunner()
            result = runner.invoke(main, ["validation-metrics", str(report_path)])

            assert result.exit_code == 0
            assert "Total sessions: 0" in result.output
            assert "Kendall Rank Correlation: N/A" in result.output

    def test_validation_metrics_single_session(self):
        """Test validation-metrics command with single session (insufficient data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            single_session_report = ValidationReport.from_results(
                session_results=[
                    SessionValidationResult(
                        session_id="session-001",
                        file="session1.jsonl",
                        predicted_score=4.2,
                        real_score=4.5,
                    )
                ],
                score_name="efficiency",
                rubrics_file="./rubrics.json",
            )
            single_session_report.to_json(report_path)

            runner = CliRunner()
            result = runner.invoke(main, ["validation-metrics", str(report_path)])

            assert result.exit_code == 0
            assert "Total sessions: 1" in result.output
            # Single session means rank_correlation is None
            assert "Kendall Rank Correlation: N/A" in result.output

    def test_validation_metrics_perfect_correlation(self):
        """Test validation-metrics command with perfect correlation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            # Create sessions with perfect positive correlation
            session_results = [
                SessionValidationResult(
                    session_id=f"s{i}",
                    file=f"s{i}.jsonl",
                    predicted_score=float(i),
                    real_score=float(i),
                )
                for i in range(1, 6)
            ]

            perfect_report = ValidationReport.from_results(
                session_results=session_results,
                score_name="test",
                rubrics_file="test.json",
            )
            perfect_report.to_json(report_path)

            runner = CliRunner()
            result = runner.invoke(main, ["validation-metrics", str(report_path)])

            assert result.exit_code == 0
            assert "Kendall Rank Correlation (τ): 1.0000" in result.output
            assert "P-value:" in result.output

    def test_validation_metrics_negative_correlation(self):
        """Test validation-metrics command with negative correlation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            # Create sessions with perfect negative correlation
            session_results = [
                SessionValidationResult(
                    session_id="s1",
                    file="s1.jsonl",
                    predicted_score=5.0,
                    real_score=1.0,
                ),
                SessionValidationResult(
                    session_id="s2",
                    file="s2.jsonl",
                    predicted_score=1.0,
                    real_score=5.0,
                ),
            ]

            negative_report = ValidationReport.from_results(
                session_results=session_results,
                score_name="test",
                rubrics_file="test.json",
            )
            negative_report.to_json(report_path)

            runner = CliRunner()
            result = runner.invoke(main, ["validation-metrics", str(report_path)])

            assert result.exit_code == 0
            assert "Kendall Rank Correlation (τ): -1.0000" in result.output
            assert "P-value:" in result.output
