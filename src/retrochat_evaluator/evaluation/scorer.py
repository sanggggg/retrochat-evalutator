"""Rubric scorer using LLM-as-a-judge."""

import asyncio
import logging
import re
from typing import Optional

from ..models.rubric import Rubric
from ..models.evaluation import RubricScore
from ..llm.gemini import GeminiClient
from ..config import EvaluationConfig

logger = logging.getLogger(__name__)


class RubricScorer:
    """Score a chat session against rubrics using LLM."""

    def __init__(
        self,
        llm_client: GeminiClient,
        config: Optional[EvaluationConfig] = None,
    ):
        """Initialize rubric scorer.

        Args:
            llm_client: Gemini client for LLM calls.
            config: Evaluation configuration.
        """
        self.llm = llm_client
        self.config = config or EvaluationConfig()

    async def score(self, rubric: Rubric, prompt: str) -> RubricScore:
        """Score a session against a single rubric.

        Args:
            rubric: The rubric being scored.
            prompt: The judge prompt to send to LLM.

        Returns:
            RubricScore with the evaluation result.
        """
        response = await self.llm.generate(
            prompt,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )

        score, reasoning = self._parse_score(response)

        # Retry once if parsing failed
        if score is None and self.config.retry_on_parse_failure:
            logger.warning(f"Failed to parse score for {rubric.id}, retrying...")
            retry_prompt = (
                prompt
                + "\n\nIMPORTANT: Please respond EXACTLY in this format:\nSCORE: [1-5]\nREASONING: [your explanation]"
            )
            response = await self.llm.generate(
                retry_prompt,
                temperature=self.config.llm_temperature,
            )
            score, reasoning = self._parse_score(response)

        # Default to middle score if still failed
        if score is None:
            logger.warning(f"Could not parse score for {rubric.id}, defaulting to 3")
            score = 3.0
            reasoning = "Unable to parse LLM response. Default score assigned."

        return RubricScore(
            rubric_id=rubric.id,
            rubric_name=rubric.name,
            score=score,
            max_score=5.0,
            reasoning=reasoning,
        )

    async def score_all(
        self,
        rubric_prompts: list[tuple[Rubric, str]],
    ) -> list[RubricScore]:
        """Score against all rubrics in parallel.

        Args:
            rubric_prompts: List of (rubric, prompt) tuples.

        Returns:
            List of RubricScore objects.
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def bounded_score(rubric: Rubric, prompt: str) -> RubricScore:
            async with semaphore:
                try:
                    return await self.score(rubric, prompt)
                except Exception as e:
                    logger.error(f"Failed to score rubric {rubric.id}: {e}")
                    return RubricScore(
                        rubric_id=rubric.id,
                        rubric_name=rubric.name,
                        score=3.0,
                        max_score=5.0,
                        reasoning=f"Scoring failed: {str(e)}",
                    )

        tasks = [bounded_score(rubric, prompt) for rubric, prompt in rubric_prompts]
        results = await asyncio.gather(*tasks)

        logger.info(f"Scored {len(results)} rubrics")
        return list(results)

    def _parse_score(self, response: str) -> tuple[Optional[float], str]:
        """Parse LLM response into score and reasoning.

        Expected format:
        SCORE: 4
        REASONING: The user demonstrated...

        Args:
            response: Raw LLM response.

        Returns:
            Tuple of (score or None, reasoning).
        """
        # Extract score
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if not score_match:
            return None, ""

        try:
            score = float(score_match.group(1))
            # Clamp to valid range
            min_score, max_score = self.config.score_scale
            score = max(min_score, min(max_score, score))
        except ValueError:
            return None, ""

        # Extract reasoning
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?:\n\n|\Z)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return score, reasoning
