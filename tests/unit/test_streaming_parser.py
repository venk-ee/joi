"""Unit tests for StreamingParser inline tag extraction and sentence chunking."""

from collections.abc import AsyncGenerator

import pytest

from joi.domain.models import SentenceChunk, TagCategory, TagEvent
from joi.services.streaming_parser import StreamingParser


async def async_token_generator(tokens: list[str]) -> AsyncGenerator[str, None]:
    for token in tokens:
        yield token


@pytest.mark.asyncio
async def test_streaming_parser_tag_and_sentence_split() -> None:
    parser = StreamingParser(turn_id=1)
    tokens = [
        "[smile] ",
        "Hey Ken. ",
        "[sit] ",
        "Are you ready to test the voice engine?",
    ]

    events: list[TagEvent | SentenceChunk] = []
    async for event in parser.parse_stream(async_token_generator(tokens)):
        events.append(event)

    assert len(events) == 4

    assert isinstance(events[0], TagEvent)
    assert events[0].name == "smile"
    assert events[0].category == TagCategory.EXPRESSION

    assert isinstance(events[1], SentenceChunk)
    assert events[1].text == "Hey Ken."
    assert events[1].sentence_idx == 0
    assert events[1].turn_id == 1

    assert isinstance(events[2], TagEvent)
    assert events[2].name == "sit"
    assert events[2].category == TagCategory.ACTION

    assert isinstance(events[3], SentenceChunk)
    assert events[3].text == "Are you ready to test the voice engine?"
    assert events[3].sentence_idx == 1


@pytest.mark.asyncio
async def test_streaming_parser_split_tokens() -> None:
    """Verifies that tags and sentences split across arbitrary token boundaries still parse cleanly."""
    parser = StreamingParser(turn_id=5)
    tokens = ["[", "c", "u", "r", "i", "o", "u", "s", "]", " What", " is that", " error", "?"]

    events: list[TagEvent | SentenceChunk] = []
    async for event in parser.parse_stream(async_token_generator(tokens)):
        events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], TagEvent)
    assert events[0].name == "curious"

    assert isinstance(events[1], SentenceChunk)
    assert events[1].text == "What is that error?"


@pytest.mark.asyncio
async def test_streaming_parser_preserves_decimals() -> None:
    """Verifies that decimal numbers like 3.14 do not trigger premature sentence splits."""
    parser = StreamingParser(turn_id=2)
    tokens = ["The constant pi is approximately 3.14159 in mathematics."]

    events: list[TagEvent | SentenceChunk] = []
    async for event in parser.parse_stream(async_token_generator(tokens)):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], SentenceChunk)
    assert events[0].text == "The constant pi is approximately 3.14159 in mathematics."


@pytest.mark.asyncio
async def test_streaming_parser_runaway_bracket_recovery() -> None:
    """Verifies that non-tag brackets like code arrays are flushed back to speech text."""
    parser = StreamingParser(turn_id=3)
    tokens = ["Check the array [1, 2, 3, 4, 5, 6, 7, 8, 9] in the code."]

    events: list[TagEvent | SentenceChunk] = []
    async for event in parser.parse_stream(async_token_generator(tokens)):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], SentenceChunk)
    assert "[1, 2" in events[0].text
