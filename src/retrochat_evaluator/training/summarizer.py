"""Rubric summarizer for consolidating extracted rubrics."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from ..models.rubric import Rubric, RubricList
from ..llm.gemini import GeminiClient
from ..utils.prompts import load_prompt_template, format_prompt

logger = logging.getLogger(__name__)


class RubricSummarizer:
    """Consolidate multiple rubric lists into a final coherent set."""

    def __init__(
        self,
        llm_client: GeminiClient,
        prompt_template_path: Path,
        prompts_dir: Optional[Path] = None,
    ):
        """Initialize rubric summarizer.

        Args:
            llm_client: Gemini client for LLM calls.
            prompt_template_path: Path to the summarizer prompt template.
            prompts_dir: Base directory for prompts.
        """
        self.llm = llm_client
        self.prompt_template = load_prompt_template(prompt_template_path, prompts_dir)

    async def summarize(self, rubric_lists: list[list[Rubric]]) -> tuple[list[Rubric], str]:
        """Consolidate rubrics into final list.

        Args:
            rubric_lists: List of rubric lists from multiple sessions.

        Returns:
            Tuple of (final rubrics list, consolidation notes).
        """
        all_rubrics = self._flatten_and_format(rubric_lists)
        prompt = format_prompt(self.prompt_template, all_rubrics=all_rubrics)

        logger.info(f"Summarizing {sum(len(r) for r in rubric_lists)} rubrics from extraction")
        response = await self.llm.generate(prompt)

        return self._parse_final_rubrics(response)

    def _flatten_and_format(self, rubric_lists: list[list[Rubric]]) -> str:
        """Combine all rubrics into formatted string for prompt.

        Args:
            rubric_lists: List of rubric lists from multiple sessions.

        Returns:
            Formatted string with all rubrics.
        """
        formatted_parts = []

        for session_idx, rubrics in enumerate(rubric_lists, 1):
            if not rubrics:
                continue

            formatted_parts.append(f"=== Session {session_idx} Rubrics ===")
            for rubric in rubrics:
                formatted_parts.append(
                    f"""
Name: {rubric.name}
Description: {rubric.description}
Scoring Criteria: {rubric.scoring_criteria}
Evidence: {rubric.evidence or 'N/A'}
---"""
                )

        return "\n".join(formatted_parts)

    def _parse_final_rubrics(self, response: str) -> tuple[list[Rubric], str]:
        """Parse LLM response into final Rubric list.

        Args:
            response: Raw LLM response.

        Returns:
            Tuple of (list of Rubric objects, consolidation notes).
        """
        # Try to find JSON object in response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.warning("No JSON object found in summarizer response")
            return [], ""

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse summarizer JSON: {e}")
            return [], ""

        rubrics_data = data.get("rubrics", [])
        consolidation_notes = data.get("consolidation_notes", "")

        rubrics = []
        for i, rdata in enumerate(rubrics_data):
            try:
                # Use provided ID or generate sequential one
                rubric_id = rdata.get("id", f"rubric_{i+1:03d}")
                # Normalize ID format
                if not rubric_id.startswith("rubric_"):
                    rubric_id = f"rubric_{i+1:03d}"

                rubric = Rubric(
                    id=rubric_id,
                    name=rdata.get("name", f"Rubric {i+1}"),
                    description=rdata.get("description", ""),
                    scoring_criteria=rdata.get("scoring_criteria", ""),
                    weight=rdata.get("weight", 1.0),
                )
                rubrics.append(rubric)
            except Exception as e:
                logger.warning(f"Failed to create final rubric {i}: {e}")

        logger.info(f"Summarized into {len(rubrics)} final rubrics")
        return rubrics, consolidation_notes
