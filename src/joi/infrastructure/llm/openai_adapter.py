"""Generic OpenAI-compatible streaming LLM adapter.

Works with any provider implementing the OpenAI /v1/chat/completions specification:
- Local endpoints: Ollama, vLLM, LM Studio, llama.cpp server, LocalAI
- Cloud endpoints: OpenRouter, OpenAI, Groq, Together, Venice, DeepSeek, Mistral
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import APIConnectionError, APIError, AsyncOpenAI, AsyncStream, RateLimitError

from joi.core.exceptions import LLMStreamError

logger = logging.getLogger("joi.llm")


class OpenAILLMAdapter:
    """Generic adapter for any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_base: str,
        api_key: str = "",
        model_name: str = "default",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_base = api_base
        # Default placeholder key for local servers that don't enforce authentication
        self.api_key = api_key if api_key.strip() else "not-needed"
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            default_headers=extra_headers or {},
        )

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streams text tokens from any OpenAI-compatible completions endpoint.

        Aborts immediately when cancel_event is triggered.
        """
        try:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens

            response = await self._client.chat.completions.create(**kwargs)

            if not isinstance(response, AsyncStream):
                logger.error("Expected AsyncStream from OpenAI completions, got %s", type(response))
                return

            async for chunk in response:
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("LLM completion stream cancelled by client barge-in.")
                    return

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except (APIError, APIConnectionError, RateLimitError) as exc:
            if cancel_event is not None and cancel_event.is_set():
                return
            logger.error("OpenAI-compatible LLM stream error (%s): %s", self.api_base, exc)
            raise LLMStreamError(f"LLM streaming failed on {self.api_base}: {exc}") from exc
