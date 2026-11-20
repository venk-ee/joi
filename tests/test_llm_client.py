import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from joi.llm.client import LLMClient


def create_mock_chunk(text):
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=text))]
    return chunk

@pytest.mark.asyncio
async def test_stream_tokens_cancellation():
    client = LLMClient(api_key="fake-key", model_name="fake-model")

    # Fake stream of 3 chunks
    chunks = [create_mock_chunk("First"), create_mock_chunk("Second"),
    create_mock_chunk("Third")]

    async def fake_stream():
        for c in chunks:
            yield c

    client.client.chat.completions.create = AsyncMock(return_value=fake_stream())

    cancel_event = asyncio.Event()
    tokens = []

    async for token in client.stream_tokens([{"role": "user", "content": "hi"}],
    cancel_event=cancel_event):
        tokens.append(token)
        cancel_event.set()  # Simulate user interrupting immediately!

    # Only "First" was yielded because cancel_event stopped the loop!
    assert tokens == ["First"]


@pytest.mark.asyncio
async def test_stream_tokens_normal():
    client = LLMClient(api_key="fake-key", model_name="fake-model")
    chunks = [create_mock_chunk("Hello"), create_mock_chunk(" world"),
    create_mock_chunk("!")]

    async def fake_stream():
        for c in chunks:
            yield c

    client.client.chat.completions.create = AsyncMock(return_value=fake_stream())

    tokens = [t async for t in client.stream_tokens([{"role": "user", "content":
    "hi"}])]
    assert tokens == ["Hello", " world", "!"]
