"""Rubric extractor from chat sessions."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

from ..models.rubric import Rubric
from ..models.chat_session import ChatSession
from ..llm.gemini import GeminiClient
from ..utils.prompts import load_prompt_template, format_prompt

logger = logging.getLogger(__name__)


class RubricExtractor:
    """Extract rubrics from a single chat session using LLM."""

    def __init__(
        self,
        llm_client: GeminiClient,
        prompt_template_path: Path,
        prompts_dir: Optional[Path] = None,
    ):
        """Initialize rubric extractor.

        Args:
            llm_client: Gemini client for LLM calls.
            prompt_template_path: Path to the extractor prompt template.
            prompts_dir: Base directory for prompts.
        """
        self.llm = llm_client
        self.prompt_template = load_prompt_template(prompt_template_path, prompts_dir)

    async def extract(self, session: ChatSession) -> list[Rubric]:
        """Extract rubrics from a chat session.

        Args:
            session: Chat session to analyze.

        Returns:
            List of extracted Rubric objects.
        """
        formatted_session = session.format_for_prompt()
        prompt = format_prompt(self.prompt_template, chat_session=formatted_session)

        logger.debug(f"Extracting rubrics from session {session.session_id}")
        response = await self.llm.generate(prompt)

        return self._parse_rubrics(response, session.session_id)

    async def extract_batch(
        self,
        sessions: list[ChatSession],
        max_concurrent: int = 5,
    ) -> list[list[Rubric]]:
        """Extract rubrics from multiple sessions in parallel.

        Args:
            sessions: List of chat sessions to analyze.
            max_concurrent: Maximum concurrent LLM calls.

        Returns:
            List of rubric lists, one per session.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_extract(session: ChatSession) -> list[Rubric]:
            async with semaphore:
                try:
                    return await self.extract(session)
                except Exception as e:
                    logger.error(f"Failed to extract from session {session.session_id}: {e}")
                    return []

        tasks = [bounded_extract(session) for session in sessions]
        results = await asyncio.gather(*tasks)

        total_rubrics = sum(len(r) for r in results)
        logger.info(f"Extracted {total_rubrics} rubrics from {len(sessions)} sessions")

        return list(results)

    def _parse_rubrics(self, response: str, session_id: str) -> list[Rubric]:
        """Parse LLM response into Rubric objects.

        Args:
            response: Raw LLM response.
            session_id: Session ID for logging.

        Returns:
            List of parsed Rubric objects.
        """
        # Try to find JSON array in response
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            logger.warning(f"No JSON array found in response for session {session_id}")
            return []

        try:
            rubrics_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for session {session_id}: {e}")
            return []

        rubrics = []
        for i, data in enumerate(rubrics_data):
            try:
                rubric = Rubric(
                    id=f"extracted_{session_id[:8]}_{i:03d}",
                    name=data.get("name", f"Rubric {i+1}"),
                    description=data.get("description", ""),
                    scoring_criteria=data.get("scoring_criteria", ""),
                    evidence=data.get("evidence"),
                )
                rubrics.append(rubric)
            except Exception as e:
                logger.warning(f"Failed to create rubric {i} for session {session_id}: {e}")

        logger.debug(f"Parsed {len(rubrics)} rubrics from session {session_id}")
        return rubrics
