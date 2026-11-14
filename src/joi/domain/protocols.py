"""Abstract protocols (ports) defining boundaries between services and adapters."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Protocol

from joi.domain.models import MemoryFact


class TTSEngine(Protocol):
    """Port for speech synthesis engines (Kokoro, Cartesia, etc.)."""

    async def warmup(self) -> None:
        """Warms up the underlying inference session on server startup."""
        ...

    def synthesize(
        self,
        text: str,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesizes text into 24kHz 16-bit linear PCM audio chunks.

        Must respect cancel_event between chunks to allow instant barge-in.
        """
        ...


class STTEngine(Protocol):
    """Port for speech-to-text engines (Faster-Whisper, Deepgram, etc.)."""

    async def warmup(self) -> None:
        """Warms up the underlying STT model on server startup."""
        ...

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribes raw PCM speech audio bytes to text."""
        ...


class LLMGateway(Protocol):
    """Port for OpenAI-compatible streaming LLM providers."""

    def stream_completion(
        self,
        messages: list[dict[str, str]],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streams tokens from the LLM provider, aborting if cancel_event is set."""
        ...


class MemoryRepository(Protocol):
    """Port for episodic memory storage and retrieval."""

    async def store_fact(self, fact: MemoryFact) -> None:
        """Persists a new user fact to episodic storage."""
        ...

    async def search_relevant_facts(self, query: str, limit: int = 5) -> list[MemoryFact]:
        """Searches top relevant facts via BM25 lexical ranking."""
        ...
