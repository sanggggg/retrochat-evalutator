"""Gemini LLM client using LangChain."""

import asyncio
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from ..config import LLMConfig, RateLimiterConfig

logger = logging.getLogger(__name__)

# Global shared rate limiter instance
_shared_rate_limiter: Optional[InMemoryRateLimiter] = None


def get_shared_rate_limiter(config: Optional[RateLimiterConfig] = None) -> InMemoryRateLimiter:
    """Get or create the shared rate limiter instance.

    Args:
        config: Rate limiter configuration. Only used on first call to create the instance.

    Returns:
        The shared InMemoryRateLimiter instance.
    """
    global _shared_rate_limiter
    if _shared_rate_limiter is None:
        config = config or RateLimiterConfig()
        _shared_rate_limiter = InMemoryRateLimiter(
            requests_per_second=config.requests_per_second,
            check_every_n_seconds=config.check_every_n_seconds,
            max_bucket_size=config.max_bucket_size,
        )
        logger.info(
            f"Created shared rate limiter: {config.requests_per_second} req/s, "
            f"max_bucket_size={config.max_bucket_size}"
        )
    return _shared_rate_limiter


def reset_shared_rate_limiter() -> None:
    """Reset the shared rate limiter (useful for testing)."""
    global _shared_rate_limiter
    _shared_rate_limiter = None


class GeminiClient:
    """Client for Google Gemini via LangChain."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
    ):
        """Initialize Gemini client.

        Args:
            config: LLM configuration. Uses defaults if not provided.
            rate_limiter_config: Rate limiter configuration. Uses shared rate limiter.
        """
        self.config = config or LLMConfig()

        if not self.config.api_key:
            raise ValueError("GOOGLE_API_KEY is required. Set it in .env or pass via config.")

        # Get the shared rate limiter (created on first call)
        self._rate_limiter = get_shared_rate_limiter(rate_limiter_config)

        self._client = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            google_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            rate_limiter=self._rate_limiter,
        )

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The generated text response.
        """
        # Create a new client with overridden settings if needed
        client = self._client
        if temperature is not None or max_tokens is not None:
            client = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                google_api_key=self.config.api_key,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_output_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                rate_limiter=self._rate_limiter,
            )

        message = HumanMessage(content=prompt)

        for attempt in range(self.config.max_retries):
            try:
                response = await client.ainvoke([message])
                return response.content
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

    async def generate_batch(
        self,
        prompts: list[str],
        max_concurrent: int = 5,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """Generate responses for multiple prompts with concurrency control.

        Args:
            prompts: List of prompts to process.
            max_concurrent: Maximum concurrent LLM calls.
            temperature: Override default temperature.

        Returns:
            List of generated responses in same order as prompts.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_generate(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt, temperature=temperature)

        tasks = [bounded_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def generate_sync(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Synchronous wrapper for generate.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Override default temperature.

        Returns:
            The generated text response.
        """
        return asyncio.run(self.generate(prompt, temperature=temperature))
