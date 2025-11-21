"""Tests for training module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrochat_evaluator.training.loader import DatasetLoader, DatasetManifest
from retrochat_evaluator.training.extractor import RubricExtractor
from retrochat_evaluator.training.summarizer import RubricSummarizer
from retrochat_evaluator.training.trainer import Trainer
from retrochat_evaluator.models.rubric import Rubric
from retrochat_evaluator.models.chat_session import ChatSession
from retrochat_evaluator.config import TrainingConfig


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_load_manifest(self, fixtures_dir: Path, mock_manifest_path: Path):
        """Test loading dataset manifest."""
        loader = DatasetLoader(fixtures_dir, mock_manifest_path)
        manifest = loader.load_manifest()

        assert len(manifest.sessions) == 3
        assert manifest.sessions[0].file == "mock_session.jsonl"
        assert manifest.sessions[0].score == 4.5

    def test_filter_by_score(self, fixtures_dir: Path, mock_manifest_path: Path):
        """Test filtering sessions by score threshold."""
        loader = DatasetLoader(fixtures_dir, mock_manifest_path)

        # Filter with threshold 4.0
        qualified = loader.filter_by_score(4.0)
        assert len(qualified) == 2  # Two sessions with score >= 4.0

        # Filter with threshold 4.5
        qualified = loader.filter_by_score(4.5)
        assert len(qualified) == 1  # Only one session with score >= 4.5

    def test_load_session(self, fixtures_dir: Path, mock_manifest_path: Path):
        """Test loading individual session."""
        loader = DatasetLoader(fixtures_dir, mock_manifest_path)
        manifest = loader.load_manifest()

        session = loader.load_session(manifest.sessions[0])
        assert isinstance(session, ChatSession)
        assert session.session_id == "test-session-001"

    def test_get_session_count(self, fixtures_dir: Path, mock_manifest_path: Path):
        """Test getting total session count."""
        loader = DatasetLoader(fixtures_dir, mock_manifest_path)
        count = loader.get_session_count()
        assert count == 3


class TestRubricExtractor:
    """Tests for RubricExtractor."""

    @pytest.fixture
    def extractor_prompt_path(self) -> Path:
        """Create temporary extractor prompt."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Analyze session:\n{chat_session}\n\nExtract rubrics as JSON array.")
            return Path(f.name)

    @pytest.mark.asyncio
    async def test_extract(
        self,
        extractor_prompt_path: Path,
        mock_llm_client: MagicMock,
        sample_chat_session: ChatSession,
        mock_extractor_response: str,
    ):
        """Test extracting rubrics from a session."""
        mock_llm_client.generate.return_value = mock_extractor_response

        extractor = RubricExtractor(
            llm_client=mock_llm_client,
            prompt_template_path=extractor_prompt_path,
        )

        rubrics = await extractor.extract(sample_chat_session)

        assert len(rubrics) == 2
        assert all(isinstance(r, Rubric) for r in rubrics)
        assert rubrics[0].name == "Clear Requirements"
        mock_llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_batch(
        self,
        extractor_prompt_path: Path,
        mock_llm_client: MagicMock,
        sample_chat_session: ChatSession,
        mock_extractor_response: str,
    ):
        """Test batch extraction from multiple sessions."""
        mock_llm_client.generate.return_value = mock_extractor_response

        extractor = RubricExtractor(
            llm_client=mock_llm_client,
            prompt_template_path=extractor_prompt_path,
        )

        sessions = [sample_chat_session, sample_chat_session]
        rubric_lists = await extractor.extract_batch(sessions, max_concurrent=2)

        assert len(rubric_lists) == 2
        assert all(len(rl) == 2 for rl in rubric_lists)

    @pytest.mark.asyncio
    async def test_extract_handles_invalid_json(
        self,
        extractor_prompt_path: Path,
        mock_llm_client: MagicMock,
        sample_chat_session: ChatSession,
    ):
        """Test handling of invalid JSON response."""
        mock_llm_client.generate.return_value = "No JSON here, just text."

        extractor = RubricExtractor(
            llm_client=mock_llm_client,
            prompt_template_path=extractor_prompt_path,
        )

        rubrics = await extractor.extract(sample_chat_session)
        assert rubrics == []


class TestRubricSummarizer:
    """Tests for RubricSummarizer."""

    @pytest.fixture
    def summarizer_prompt_path(self) -> Path:
        """Create temporary summarizer prompt."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Consolidate rubrics:\n{all_rubrics}\n\nReturn JSON object.")
            return Path(f.name)

    @pytest.mark.asyncio
    async def test_summarize(
        self,
        summarizer_prompt_path: Path,
        mock_llm_client: MagicMock,
        sample_rubrics: list[Rubric],
        mock_summarizer_response: str,
    ):
        """Test summarizing rubrics."""
        mock_llm_client.generate.return_value = mock_summarizer_response

        summarizer = RubricSummarizer(
            llm_client=mock_llm_client,
            prompt_template_path=summarizer_prompt_path,
        )

        rubric_lists = [[sample_rubrics[0], sample_rubrics[1]], [sample_rubrics[2]]]
        final_rubrics, notes = await summarizer.summarize(rubric_lists)

        assert len(final_rubrics) == 2
        assert all(isinstance(r, Rubric) for r in final_rubrics)
        assert "rubric_001" in [r.id for r in final_rubrics]
        mock_llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_handles_invalid_json(
        self,
        summarizer_prompt_path: Path,
        mock_llm_client: MagicMock,
        sample_rubrics: list[Rubric],
    ):
        """Test handling of invalid JSON response."""
        mock_llm_client.generate.return_value = "Invalid response without JSON."

        summarizer = RubricSummarizer(
            llm_client=mock_llm_client,
            prompt_template_path=summarizer_prompt_path,
        )

        rubric_lists = [[sample_rubrics[0]]]
        final_rubrics, notes = await summarizer.summarize(rubric_lists)

        assert final_rubrics == []


class TestTrainer:
    """Tests for Trainer orchestrator."""

    @pytest.fixture
    def prompts_dir(self) -> Path:
        """Create temporary prompts directory with required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir)

            # Create extractor prompt
            (prompts_dir / "rubric_extractor.txt").write_text(
                "Extract rubrics from:\n{chat_session}"
            )
            # Create summarizer prompt
            (prompts_dir / "rubric_summarizer.txt").write_text(
                "Summarize rubrics:\n{all_rubrics}"
            )

            yield prompts_dir

    @pytest.mark.asyncio
    async def test_train_integration(
        self,
        fixtures_dir: Path,
        mock_manifest_path: Path,
        prompts_dir: Path,
        mock_extractor_response: str,
        mock_summarizer_response: str,
    ):
        """Test full training pipeline with mocked LLM."""
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            side_effect=[
                mock_extractor_response,  # First session extraction
                mock_extractor_response,  # Second session extraction
                mock_summarizer_response,  # Summarization
            ]
        )

        config = TrainingConfig(score_threshold=4.0)
        trainer = Trainer(
            dataset_dir=fixtures_dir,
            manifest_path=mock_manifest_path,
            prompts_dir=prompts_dir,
            config=config,
        )
        # Inject mock client directly
        trainer._llm_client = mock_client

        rubric_list = await trainer.train()

        assert rubric_list is not None
        assert len(rubric_list.rubrics) == 2
        assert rubric_list.training_config is not None
        assert rubric_list.training_config.sessions_used == 2

    def test_save_rubrics(
        self, fixtures_dir: Path, mock_manifest_path: Path, sample_rubric_list
    ):
        """Test saving rubrics to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "rubric_extractor.txt").write_text("{chat_session}")
            (prompts_dir / "rubric_summarizer.txt").write_text("{all_rubrics}")

            trainer = Trainer(
                dataset_dir=fixtures_dir,
                manifest_path=mock_manifest_path,
                prompts_dir=prompts_dir,
            )

            output_path = Path(tmpdir) / "rubrics.json"
            trainer.save_rubrics(sample_rubric_list, output_path)

            assert output_path.exists()
