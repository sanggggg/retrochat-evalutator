"""LLM-backed rubric summarizer with hierarchical batching."""

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
        min_rubrics: int = 5,
        max_rubrics: int = 10,
        max_batch_size: int = 100,
    ):
        """Initialize rubric summarizer.

        Args:
            llm_client: Gemini client for LLM calls.
            prompt_template_path: Path to the summarizer prompt template.
            prompts_dir: Base directory for prompts.
            min_rubrics: Minimum number of rubrics in final list.
            max_rubrics: Maximum number of rubrics in final list.
            max_batch_size: Maximum number of rubrics to send to the LLM at once.
        """
        self.llm = llm_client
        self.prompt_template = load_prompt_template(prompt_template_path, prompts_dir)
        self.min_rubrics = min_rubrics
        self.max_rubrics = max_rubrics
        self.max_batch_size = max_batch_size

    async def summarize(self, rubric_lists: list[list[Rubric]]) -> tuple[list[Rubric], str]:
        """Consolidate rubrics into final list.

        Args:
            rubric_lists: List of rubric lists from multiple sessions.

        Returns:
            Tuple of (final rubrics list, consolidation notes).
        """
        collected = self._collect_rubrics(rubric_lists)
        total_rubrics = len(collected)

        if total_rubrics == 0:
            logger.info("No rubrics provided for summarization")
            return [], ""

        if total_rubrics >= self.max_batch_size:
            logger.info(
                "Summarizing %s rubrics in batches of %s",
                total_rubrics,
                self.max_batch_size,
            )
            aggregated: RubricList = []
            notes: list[str] = []

            for batch_idx, chunk in enumerate(
                self._chunk_rubrics(collected, self.max_batch_size), 1
            ):
                logger.debug(
                    "Processing rubric chunk %s with %s entries", batch_idx, len(chunk)
                )
                chunk_rubrics, chunk_notes = await self._summarize_batch(chunk)
                aggregated.extend(chunk_rubrics)
                if chunk_notes:
                    notes.append(chunk_notes)

            if not aggregated:
                return [], "\n\n".join(notes).strip()

            final_rubrics, final_notes = await self.summarize([aggregated])
            if final_notes:
                notes.append(final_notes)

            combined_notes = "\n\n".join(notes).strip()
            return final_rubrics, combined_notes

        return await self._summarize_batch(collected)

    async def _summarize_batch(self, rubrics: RubricList) -> tuple[list[Rubric], str]:
        """Send a single batch of rubrics to the LLM for consolidation."""
        prompt = format_prompt(
            self.prompt_template,
            all_rubrics=self._format_rubrics_for_prompt([rubrics]),
            summarization_min_rubrics=self.min_rubrics,
            summarization_max_rubrics=self.max_rubrics,
        )

        logger.info("Summarizing batch of %s rubrics", len(rubrics))
        response = await self.llm.generate(prompt)

        return self._parse_final_rubrics(response)

    def _collect_rubrics(self, rubric_lists: list[list[Rubric]]) -> RubricList:
        """Flatten nested rubric lists into a single list."""
        collected: RubricList = []
        for rubrics in rubric_lists:
            if rubrics:
                collected.extend(rubrics)
        return collected

    def _chunk_rubrics(self, rubrics: RubricList, chunk_size: int) -> list[RubricList]:
        """Split rubrics into evenly-sized chunks for hierarchical summarization."""
        return [rubrics[i : i + chunk_size] for i in range(0, len(rubrics), chunk_size)]

    def _format_rubrics_for_prompt(self, rubric_lists: list[list[Rubric]]) -> str:
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
                    evidence=None,
                )
                rubrics.append(rubric)
            except Exception as e:
                logger.warning(f"Failed to create final rubric {i}: {e}")

        logger.info(f"Summarized into {len(rubrics)} final rubrics")
        return rubrics, consolidation_notes
