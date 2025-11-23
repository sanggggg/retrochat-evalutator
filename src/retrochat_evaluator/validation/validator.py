"""Validator for comparing LLM-predicted scores against real scores."""

import logging
from pathlib import Path
from typing import Optional

from ..models.rubric import Rubric, RubricList
from ..models.chat_session import ChatSession
from ..models.evaluation import EvaluationResult
from ..models.validation import (
    SessionValidationResult,
    ValidationReport,
)
from ..config import EvaluationConfig, EvaluationLLMConfig
from ..training.loader import DatasetLoader, SessionInfo
from ..evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class Validator:
    """Validate evaluation accuracy by comparing predicted vs real scores."""

    def __init__(
        self,
        dataset_dir: Path,
        manifest_path: Path,
        rubrics_path: Path,
        prompts_dir: Path,
        score_name: str = "default",
        config: Optional[EvaluationConfig] = None,
        llm_config: Optional[EvaluationLLMConfig] = None,
    ):
        """Initialize validator.

        Args:
            dataset_dir: Directory containing JSONL session files.
            manifest_path: Path to the dataset manifest JSON file.
            rubrics_path: Path to the rubrics JSON file.
            prompts_dir: Directory containing prompt templates.
            score_name: Name of the score to use for comparison.
            config: Evaluation configuration.
            llm_config: LLM configuration for evaluation.
        """
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = Path(manifest_path)
        self.rubrics_path = Path(rubrics_path)
        self.prompts_dir = Path(prompts_dir)
        self.score_name = score_name

        self.config = config or EvaluationConfig()
        self.llm_config = llm_config or EvaluationLLMConfig()

        # Initialize components
        self._loader: Optional[DatasetLoader] = None
        self._evaluator: Optional[Evaluator] = None
        self._rubric_list: Optional[RubricList] = None

    def _get_loader(self) -> DatasetLoader:
        """Get or create dataset loader."""
        if self._loader is None:
            self._loader = DatasetLoader(self.dataset_dir, self.manifest_path)
        return self._loader

    def _get_evaluator(self) -> Evaluator:
        """Get or create evaluator."""
        if self._evaluator is None:
            self._evaluator = Evaluator(
                prompts_dir=self.prompts_dir,
                config=self.config,
                llm_config=self.llm_config,
            )
        return self._evaluator

    def _get_rubrics(self) -> RubricList:
        """Get or create rubric list."""
        if self._rubric_list is None:
            self._rubric_list = RubricList.from_json(self.rubrics_path)
        return self._rubric_list

    async def validate(
        self,
        max_sessions: Optional[int] = None,
    ) -> ValidationReport:
        """Execute full validation pipeline.

        Args:
            max_sessions: Maximum number of sessions to validate.

        Returns:
            ValidationReport comparing predicted vs real scores.
        """
        # Load components
        loader = self._get_loader()
        evaluator = self._get_evaluator()
        rubric_list = self._get_rubrics()

        # Get all sessions from manifest
        manifest = loader.load_manifest()
        session_infos = manifest.sessions

        # Filter to sessions that have the specified score
        session_infos = [s for s in session_infos if s.get_score(self.score_name) is not None]

        if not session_infos:
            logger.warning(f"No sessions found with score '{self.score_name}'")
            return ValidationReport.from_results(
                session_results=[],
                score_name=self.score_name,
                rubrics_file=str(self.rubrics_path),
            )

        if max_sessions:
            session_infos = session_infos[:max_sessions]

        logger.info(
            f"Validating {len(session_infos)} sessions against {len(rubric_list.rubrics)} rubrics"
        )

        # Process each session
        session_results: list[SessionValidationResult] = []

        for i, session_info in enumerate(session_infos, 1):
            logger.info(f"Processing session {i}/{len(session_infos)}: {session_info.file}")

            try:
                result = await self._validate_session(
                    session_info=session_info,
                    rubrics=rubric_list.rubrics,
                    rubrics_version=rubric_list.version,
                    evaluator=evaluator,
                    loader=loader,
                )
                session_results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate session {session_info.file}: {e}")

        # Generate report
        report = ValidationReport.from_results(
            session_results=session_results,
            score_name=self.score_name,
            rubrics_file=str(self.rubrics_path),
        )

        logger.info(
            f"Validation complete. MAE: {report.metrics.mean_absolute_error}, "
            f"RMSE: {report.metrics.root_mean_squared_error}, "
            f"Correlation: {report.metrics.correlation}"
        )

        return report

    async def _validate_session(
        self,
        session_info: SessionInfo,
        rubrics: list[Rubric],
        rubrics_version: Optional[str],
        evaluator: Evaluator,
        loader: DatasetLoader,
    ) -> SessionValidationResult:
        """Validate a single session.

        Args:
            session_info: Session information from manifest.
            rubrics: List of rubrics to evaluate against.
            rubrics_version: Version of rubrics.
            evaluator: Evaluator instance.
            loader: Dataset loader instance.

        Returns:
            SessionValidationResult with comparison.
        """
        # Load session
        session = loader.load_session(session_info)

        # Evaluate session
        eval_result = await evaluator.evaluate(
            rubrics=rubrics,
            session=session,
            rubrics_version=rubrics_version,
        )

        # Get real score
        real_score = session_info.get_score(self.score_name)
        predicted_score = eval_result.summary.total_score

        # Calculate error metrics
        error = predicted_score - real_score
        absolute_error = abs(error)
        squared_error = error**2

        return SessionValidationResult(
            session_id=session.session_id,
            file=session_info.file,
            predicted_score=round(predicted_score, 4),
            real_score=real_score,
            error=round(error, 4),
            absolute_error=round(absolute_error, 4),
            squared_error=round(squared_error, 4),
        )

    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to JSON file.

        Args:
            report: ValidationReport to save.
            output_path: Path for output JSON file.
        """
        report.to_json(output_path)
        logger.info(f"Saved validation report to {output_path}")
