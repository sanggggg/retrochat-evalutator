"""Tests for training module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrochat_evaluator.training.loader import DatasetLoader, DatasetManifest
from retrochat_evaluator.training.extractor import RubricExtractor
from retrochat_evaluator.training.summarizer import RubricSummarizer
from retrochat_evaluator.training.semantic_summarizer import SemanticClusteringSummarizer
from retrochat_evaluator.training.trainer import Trainer
from retrochat_evaluator.models.rubric import Rubric
from retrochat_evaluator.models.chat_session import ChatSession
from retrochat_evaluator.config import TrainingConfig, SummarizationMethod


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_load_manifest(self, fixtures_dir: Path, mock_manifest_path: Path):
        """Test loading dataset manifest."""
        loader = DatasetLoader(fixtures_dir, mock_manifest_path)
        manifest = loader.load_manifest()

        assert len(manifest.sessions) == 3
        assert manifest.sessions[0].file == "mock_session.jsonl"
        assert manifest.sessions[0].scores == {"efficiency": 4.5, "quality": 4.0}
        assert manifest.sessions[0].get_score("efficiency") == 4.5

    def test_filter_by_score(self, fixtures_dir: Path, mock_manifest_path: Path):
        """Test filtering sessions by score threshold."""
        loader = DatasetLoader(fixtures_dir, mock_manifest_path)

        # Filter with threshold 4.0 on efficiency score
        qualified = loader.filter_by_score(4.0, "efficiency")
        assert len(qualified) == 2  # Two sessions with efficiency score >= 4.0

        # Filter with threshold 4.5 on efficiency score
        qualified = loader.filter_by_score(4.5, "efficiency")
        assert len(qualified) == 1  # Only one session with efficiency score >= 4.5

    def test_filter_by_percentile(self, fixtures_dir: Path, mock_manifest_path: Path):
        """Test filtering sessions by top percentile."""
        loader = DatasetLoader(fixtures_dir, mock_manifest_path)

        qualified = loader.filter_by_percentile(50.0, "efficiency")
        assert len(qualified) == 2  # Top 50% of 3 sessions => 2 sessions
        assert all(session.file != "low_score_session.jsonl" for session in qualified)

    def test_filter_by_different_score_name(self, fixtures_dir: Path, mock_manifest_path: Path):
        """Test filtering sessions by different score types."""
        loader = DatasetLoader(fixtures_dir, mock_manifest_path)

        # Filter by quality score
        qualified = loader.filter_by_score(3.9, "quality")
        assert len(qualified) == 1  # Only one session with quality score >= 3.9

        # Filter by non-existent score name
        qualified = loader.filter_by_score(4.0, "nonexistent")
        assert len(qualified) == 0  # No sessions have this score type

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
            (prompts_dir / "rubric_summarizer.txt").write_text("Summarize rubrics:\n{all_rubrics}")

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

        config = TrainingConfig(score_top_percentile=50.0, score_name="efficiency")
        trainer = Trainer(
            dataset_dir=fixtures_dir,
            manifest_path=mock_manifest_path,
            prompts_dir=prompts_dir,
            config=config,
        )
        # Inject mock clients directly
        trainer._extraction_llm_client = mock_client
        trainer._summarization_llm_client = mock_client

        rubric_list, raw_rubrics_map, _ = await trainer.train()

        assert rubric_list is not None
        assert len(rubric_list.rubrics) == 2
        assert rubric_list.training_config is not None
        assert rubric_list.training_config.sessions_used == 2
        assert isinstance(raw_rubrics_map, dict)

    @pytest.mark.asyncio
    async def test_save_rubrics(self, fixtures_dir: Path, mock_manifest_path: Path, sample_rubric_list):
        """Test saving rubrics to folder structure."""
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

            output_dir = Path(tmpdir) / "output"
            raw_rubrics_map = {
                "session1": sample_rubric_list.rubrics[:1],
                "session2": sample_rubric_list.rubrics[1:],
            }
            result_folder = await trainer.save_rubrics(
                sample_rubric_list, output_dir, raw_rubrics_map, validate=False
            )

            assert result_folder.exists()
            assert (result_folder / "rubrics.json").exists()
            assert (result_folder / "metadata.json").exists()
            assert (result_folder / "raw-rubrics.json").exists()


class TestSemanticClusteringSummarizer:
    """Tests for SemanticClusteringSummarizer."""

    @pytest.fixture
    def similar_rubrics(self) -> list[list[Rubric]]:
        """Create rubric lists with semantically similar items for clustering tests."""
        # Group 1: Communication clarity related (should cluster together)
        rubric_1a = Rubric(
            id="r1a",
            name="Clear Communication",
            description="The user provides clear and specific requirements",
            scoring_criteria="1: Vague, 5: Excellent",
        )
        rubric_1b = Rubric(
            id="r1b",
            name="Communication Clarity",
            description="User clearly states their needs and expectations",
            scoring_criteria="1: Unclear, 5: Very clear",
        )
        rubric_1c = Rubric(
            id="r1c",
            name="Requirement Clarity",
            description="Requirements are communicated clearly and precisely",
            scoring_criteria="1: Ambiguous, 5: Crystal clear",
        )

        # Group 2: Efficiency related (should cluster together)
        rubric_2a = Rubric(
            id="r2a",
            name="Task Efficiency",
            description="The user guides the AI efficiently toward the solution",
            scoring_criteria="1: Inefficient, 5: Optimal",
        )
        rubric_2b = Rubric(
            id="r2b",
            name="Interaction Efficiency",
            description="Efficient interaction with minimal unnecessary exchanges",
            scoring_criteria="1: Many redundant steps, 5: Direct path",
        )

        # Group 3: Context provision (should cluster together)
        rubric_3a = Rubric(
            id="r3a",
            name="Context Provision",
            description="The user provides necessary context and background information",
            scoring_criteria="1: No context, 5: Comprehensive",
        )
        rubric_3b = Rubric(
            id="r3b",
            name="Background Information",
            description="Adequate background context is provided for the task",
            scoring_criteria="1: Missing context, 5: Complete background",
        )

        # Simulate extraction from multiple sessions
        return [
            [rubric_1a, rubric_2a, rubric_3a],  # Session 1
            [rubric_1b, rubric_2b, rubric_3b],  # Session 2
            [rubric_1c],  # Session 3
        ]

    @pytest.fixture
    def diverse_rubrics(self) -> list[list[Rubric]]:
        """Create rubric lists with diverse, non-overlapping items."""
        return [
            [
                Rubric(
                    id="div1",
                    name="Code Quality",
                    description="The code follows best practices and is well-structured",
                    scoring_criteria="1: Poor, 5: Excellent",
                ),
            ],
            [
                Rubric(
                    id="div2",
                    name="Error Handling",
                    description="Proper error handling and edge case coverage",
                    scoring_criteria="1: None, 5: Comprehensive",
                ),
            ],
            [
                Rubric(
                    id="div3",
                    name="Documentation",
                    description="Clear documentation and code comments",
                    scoring_criteria="1: Missing, 5: Thorough",
                ),
            ],
        ]

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings that simulate semantic similarity."""
        import numpy as np

        # Create embeddings where similar rubrics have similar vectors
        # Group 1 (communication/clarity): similar embeddings
        # Group 2 (efficiency): similar embeddings
        # Group 3 (context): similar embeddings
        embeddings = {
            # Group 1: Communication clarity - vectors close together
            0: np.array([0.9, 0.1, 0.0]),  # Clear Communication
            1: np.array([0.85, 0.15, 0.0]),  # Communication Clarity
            6: np.array([0.88, 0.12, 0.0]),  # Requirement Clarity
            # Group 2: Efficiency - vectors close together
            2: np.array([0.1, 0.9, 0.0]),  # Task Efficiency
            3: np.array([0.15, 0.85, 0.0]),  # Interaction Efficiency
            # Group 3: Context - vectors close together
            4: np.array([0.0, 0.1, 0.9]),  # Context Provision
            5: np.array([0.0, 0.15, 0.85]),  # Background Information
        }
        return embeddings

    @pytest.mark.asyncio
    async def test_summarize_clusters_similar_rubrics(self, similar_rubrics, mock_embeddings):
        """Test that similar rubrics are clustered together."""
        import numpy as np

        summarizer = SemanticClusteringSummarizer(
            embedding_model="models/text-embedding-004",
            similarity_threshold=0.6,  # Lower threshold for testing
            min_rubrics=2,
            max_rubrics=5,
        )

        # Create mock embedding array in order
        mock_embedding_array = np.array([mock_embeddings[i] for i in range(7)])

        with patch.object(
            summarizer, "_generate_embeddings", new=AsyncMock(return_value=mock_embedding_array)
        ):
            final_rubrics, notes = await summarizer.summarize(similar_rubrics)

        # Should cluster into roughly 3 groups
        assert len(final_rubrics) >= 2
        assert len(final_rubrics) <= 5
        assert all(isinstance(r, Rubric) for r in final_rubrics)
        assert "Semantic clustering" in notes

    @pytest.mark.asyncio
    async def test_summarize_respects_max_rubrics(self, similar_rubrics, mock_embeddings):
        """Test that max_rubrics limit is respected."""
        import numpy as np

        summarizer = SemanticClusteringSummarizer(
            similarity_threshold=0.5,
            min_rubrics=1,
            max_rubrics=2,
        )

        mock_embedding_array = np.array([mock_embeddings[i] for i in range(7)])

        with patch.object(
            summarizer, "_generate_embeddings", new=AsyncMock(return_value=mock_embedding_array)
        ):
            final_rubrics, notes = await summarizer.summarize(similar_rubrics)

        assert len(final_rubrics) <= 2

    @pytest.mark.asyncio
    async def test_summarize_respects_min_rubrics(self, similar_rubrics):
        """Test that min_rubrics is respected when enough clusters exist."""
        import numpy as np

        summarizer = SemanticClusteringSummarizer(
            similarity_threshold=0.99,  # Very high threshold = each item is its own cluster
            min_rubrics=3,
            max_rubrics=10,
        )

        # Create very distinct embeddings so each item is its own cluster
        distinct_embeddings = np.eye(7)  # 7 rubrics, each with unique direction

        with patch.object(
            summarizer, "_generate_embeddings", new=AsyncMock(return_value=distinct_embeddings)
        ):
            final_rubrics, notes = await summarizer.summarize(similar_rubrics)

        # With very high threshold, should have at least min_rubrics clusters
        assert len(final_rubrics) >= 3

    @pytest.mark.asyncio
    async def test_summarize_empty_input(self):
        """Test handling of empty input."""
        summarizer = SemanticClusteringSummarizer()

        final_rubrics, notes = await summarizer.summarize([])
        assert final_rubrics == []
        assert "No rubrics" in notes

    @pytest.mark.asyncio
    async def test_summarize_single_rubric(self):
        """Test handling of single rubric input (below min_rubrics, no clustering)."""
        # Set min_rubrics=5 so single rubric triggers deduplication path (no embedding)
        summarizer = SemanticClusteringSummarizer(min_rubrics=5, max_rubrics=10)
        single_rubric = Rubric(
            id="single",
            name="Test Rubric",
            description="A single test rubric",
            scoring_criteria="1-5 scale",
        )

        # With min_rubrics=5 and only 1 rubric, it should skip clustering
        final_rubrics, notes = await summarizer.summarize([[single_rubric]])

        assert len(final_rubrics) == 1
        assert final_rubrics[0].name == "Test Rubric"
        assert "no clustering needed" in notes.lower()

    @pytest.mark.asyncio
    async def test_summarize_assigns_sequential_ids(self, similar_rubrics, mock_embeddings):
        """Test that output rubrics have sequential IDs."""
        import numpy as np

        summarizer = SemanticClusteringSummarizer(
            similarity_threshold=0.5,
            min_rubrics=2,
            max_rubrics=5,
        )

        mock_embedding_array = np.array([mock_embeddings[i] for i in range(7)])

        with patch.object(
            summarizer, "_generate_embeddings", new=AsyncMock(return_value=mock_embedding_array)
        ):
            final_rubrics, notes = await summarizer.summarize(similar_rubrics)

        for i, rubric in enumerate(final_rubrics):
            assert rubric.id == f"rubric_{i+1:03d}"

    @pytest.mark.asyncio
    async def test_summarize_diverse_rubrics_preserved(self, diverse_rubrics):
        """Test that diverse rubrics are preserved (not incorrectly merged)."""
        import numpy as np

        summarizer = SemanticClusteringSummarizer(
            similarity_threshold=0.8,  # High threshold - only very similar items cluster
            min_rubrics=1,
            max_rubrics=10,
        )

        # Create very distinct embeddings for diverse rubrics
        distinct_embeddings = np.eye(3)  # 3 diverse rubrics

        with patch.object(
            summarizer, "_generate_embeddings", new=AsyncMock(return_value=distinct_embeddings)
        ):
            final_rubrics, notes = await summarizer.summarize(diverse_rubrics)

        # Diverse rubrics should mostly remain separate
        assert len(final_rubrics) >= 2

    @pytest.mark.asyncio
    async def test_summarize_notes_contain_metadata(self, similar_rubrics, mock_embeddings):
        """Test that consolidation notes contain useful metadata."""
        import numpy as np

        summarizer = SemanticClusteringSummarizer(
            embedding_model="models/text-embedding-004",
            similarity_threshold=0.75,
        )

        mock_embedding_array = np.array([mock_embeddings[i] for i in range(7)])

        with patch.object(
            summarizer, "_generate_embeddings", new=AsyncMock(return_value=mock_embedding_array)
        ):
            final_rubrics, notes = await summarizer.summarize(similar_rubrics)

        assert "Input:" in notes
        assert "Output:" in notes
        assert "DBSCAN" in notes
        assert "text-embedding-004" in notes


class TestTrainerWithSemanticClustering:
    """Tests for Trainer with semantic clustering method."""

    @pytest.fixture
    def prompts_dir_with_files(self) -> Path:
        """Create temporary prompts directory with required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir)

            # Create extractor prompt
            (prompts_dir / "rubric_extractor.txt").write_text(
                "Extract rubrics from:\n{chat_session}"
            )
            # Create summarizer prompt (not used for semantic clustering but required)
            (prompts_dir / "rubric_summarizer.txt").write_text("Summarize rubrics:\n{all_rubrics}")

            yield prompts_dir

    @pytest.mark.asyncio
    async def test_train_with_semantic_clustering(
        self,
        fixtures_dir: Path,
        mock_manifest_path: Path,
        prompts_dir_with_files: Path,
        mock_extractor_response: str,
    ):
        """Test training pipeline using semantic clustering method."""
        import numpy as np

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            side_effect=[
                mock_extractor_response,  # First session extraction
                mock_extractor_response,  # Second session extraction
                # Note: No summarizer call - semantic clustering doesn't use LLM
            ]
        )

        config = TrainingConfig(
            score_top_percentile=50.0,
            score_name="efficiency",
            summarization_method=SummarizationMethod.SEMANTIC_CLUSTERING,
            similarity_threshold=0.6,
            min_rubrics=1,
            max_rubrics=5,
        )
        trainer = Trainer(
            dataset_dir=fixtures_dir,
            manifest_path=mock_manifest_path,
            prompts_dir=prompts_dir_with_files,
            config=config,
        )
        # Inject mock client for extraction only
        trainer._extraction_llm_client = mock_client

        # Mock the embeddings generation for semantic clustering
        # Each extraction returns 2 rubrics, so 4 total rubrics
        mock_embeddings = np.array(
            [
                [0.9, 0.1, 0.0],  # Rubric 1 from session 1
                [0.85, 0.15, 0.0],  # Rubric 2 from session 1
                [0.1, 0.9, 0.0],  # Rubric 1 from session 2
                [0.15, 0.85, 0.0],  # Rubric 2 from session 2
            ]
        )

        with patch(
            "retrochat_evaluator.training.semantic_summarizer.SemanticClusteringSummarizer._generate_embeddings",
            new=AsyncMock(return_value=mock_embeddings),
        ):
            rubric_list, raw_rubrics_map, _ = await trainer.train()

        assert rubric_list is not None
        assert len(rubric_list.rubrics) >= 1
        assert rubric_list.training_config is not None
        assert isinstance(raw_rubrics_map, dict)
        # Semantic clustering should have been called (2 extractions, no summarization LLM call)
        assert mock_client.generate.call_count == 2
