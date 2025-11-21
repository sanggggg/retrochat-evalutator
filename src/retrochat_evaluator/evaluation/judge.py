"""Judge prompt generator for evaluation."""

import logging
from pathlib import Path
from typing import Optional

from ..models.rubric import Rubric
from ..models.chat_session import ChatSession
from ..utils.prompts import load_prompt_template, format_prompt

logger = logging.getLogger(__name__)


class JudgePromptGenerator:
    """Generate evaluation prompts from template and rubrics."""

    def __init__(
        self,
        template_path: Path,
        prompts_dir: Optional[Path] = None,
    ):
        """Initialize judge prompt generator.

        Args:
            template_path: Path to the judge prompt template.
            prompts_dir: Base directory for prompts.
        """
        self.template = load_prompt_template(template_path, prompts_dir)

    def generate(self, rubric: Rubric, session: ChatSession) -> str:
        """Generate a judge prompt for a specific rubric.

        Args:
            rubric: The rubric to evaluate against.
            session: The chat session to evaluate.

        Returns:
            Formatted judge prompt.
        """
        formatted_session = session.format_for_prompt()

        return format_prompt(
            self.template,
            rubric_name=rubric.name,
            rubric_description=rubric.description,
            scoring_criteria=rubric.scoring_criteria,
            chat_session=formatted_session,
        )

    def generate_all(
        self,
        rubrics: list[Rubric],
        session: ChatSession,
    ) -> list[tuple[Rubric, str]]:
        """Generate judge prompts for all rubrics.

        Args:
            rubrics: List of rubrics to evaluate.
            session: The chat session to evaluate.

        Returns:
            List of (rubric, prompt) tuples.
        """
        result = []
        for rubric in rubrics:
            prompt = self.generate(rubric, session)
            result.append((rubric, prompt))

        logger.debug(f"Generated {len(result)} judge prompts for session {session.session_id}")
        return result
