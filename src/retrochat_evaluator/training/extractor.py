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
        
        if not response:
            logger.warning(f"Empty response from LLM for session {session.session_id}")
            return []
        
        if not isinstance(response, str):
            logger.warning(
                f"Unexpected response type {type(response).__name__} for session {session.session_id}"
            )
            return []

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
                    logger.error(
                        f"Failed to extract from session {session.session_id}: "
                        f"{type(e).__name__}: {e}",
                        exc_info=True
                    )
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
        # First, try to extract JSON from markdown code blocks if present
        json_text = response
        code_block_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", response, re.DOTALL)
        if code_block_match:
            json_text = code_block_match.group(1)
            logger.debug(f"Found JSON in code block for session {session_id}")
        else:
            # Try to find JSON array in response
            json_match = re.search(r"\[[\s\S]*?\]", response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                logger.warning(
                    f"No JSON array found in response for session {session_id}. "
                    f"Response preview: {response[:200]}..."
                )
                return []

        try:
            rubrics_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON for session {session_id}: {e}\n"
                f"JSON text preview: {json_text[:500]}..."
            )
            return []

        # Validate that we got a list
        if not isinstance(rubrics_data, list):
            logger.warning(
                f"Expected JSON array but got {type(rubrics_data).__name__} "
                f"for session {session_id}"
            )
            return []

        rubrics = []
        for i, data in enumerate(rubrics_data):
            try:
                # Validate that each item is a dict
                if not isinstance(data, dict):
                    logger.warning(
                        f"Expected dict for rubric {i} in session {session_id}, "
                        f"got {type(data).__name__}: {data}"
                    )
                    continue

                rubric = Rubric(
                    id=f"extracted_{session_id[:8]}_{i:03d}",
                    name=data.get("name", f"Rubric {i+1}"),
                    description=data.get("description", ""),
                    scoring_criteria=data.get("scoring_criteria", ""),
                    evidence=data.get("evidence"),
                )
                rubrics.append(rubric)
            except Exception as e:
                logger.warning(
                    f"Failed to create rubric {i} for session {session_id}: {e}\n"
                    f"Data: {data}"
                )

        logger.debug(f"Parsed {len(rubrics)} rubrics from session {session_id}")
        return rubrics
