"""Unit tests for TurnOrchestrator verifying full-duplex turn lifecycle, barge-in, and mocks."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from joi.domain.events import unpack_audio_frame
from joi.services.orchestrator import TurnOrchestrator


class MockLLMGateway:
    """Mock LLM returning deterministic token sequences."""

    def __init__(self, token_batches: list[str]) -> None:
        self.token_batches = token_batches

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        for token in self.token_batches:
            if cancel_event is not None and cancel_event.is_set():
                return
            yield token


class MockTTSEngine:
    """Mock TTS returning instant dummy PCM audio frames."""

    def __init__(self) -> None:
        self.warmup_called = False

    async def warmup(self) -> None:
        self.warmup_called = True

    async def synthesize(
        self,
        text: str,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[bytes, None]:
        if cancel_event is not None and cancel_event.is_set():
            return
        yield b"\x00\x01\x02\x03" * 10


class MockSTTEngine:
    """Mock STT returning predetermined transcripts."""

    async def warmup(self) -> None:
        pass

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        return "Hey Joi, sit down."


@pytest.mark.asyncio
async def test_orchestrator_complete_turn() -> None:
    sent_json: list[dict[str, Any]] = []
    sent_bytes: list[bytes] = []

    async def mock_send_json(data: dict[str, Any]) -> None:
        sent_json.append(data)

    async def mock_send_bytes(data: bytes) -> None:
        sent_bytes.append(data)

    llm = MockLLMGateway([
        "[smile] ",
        "Hello Ken. ",
        "[sit] ",
        "Let's write some clean code today.",
    ])
    tts = MockTTSEngine()
    stt = MockSTTEngine()

    orchestrator = TurnOrchestrator(
        llm_gateway=llm,
        tts_engine=tts,
        stt_engine=stt,
        send_json=mock_send_json,
        send_bytes=mock_send_bytes,
    )

    await orchestrator.handle_user_message("Hello!")

    # Wait briefly for active turn task to finish
    if orchestrator._active_task:
        await orchestrator._active_task

    # Validate JSON message sequence:
    # 1. turn_start
    # 2. expression (smile)
    # 3. action (sit)
    # 4. turn_end
    types = [msg["type"] for msg in sent_json]
    assert types == ["turn_start", "expression", "action", "turn_end"]

    assert sent_json[1]["data"]["blendshape"] == "smile"
    assert sent_json[2]["data"]["animation"] == "sit"

    # Validate binary audio frames: 2 sentences -> 2 frames
    assert len(sent_bytes) == 2
    turn_id_0, seq_0, pcm_0 = unpack_audio_frame(sent_bytes[0])
    assert turn_id_0 == 1
    assert seq_0 == 0
    assert len(pcm_0) == 40

    turn_id_1, seq_1, pcm_1 = unpack_audio_frame(sent_bytes[1])
    assert turn_id_1 == 1
    assert seq_1 == 1
    assert len(pcm_1) == 40


@pytest.mark.asyncio
async def test_orchestrator_barge_in() -> None:
    sent_json: list[dict[str, Any]] = []
    sent_bytes: list[bytes] = []

    async def mock_send_json(data: dict[str, Any]) -> None:
        sent_json.append(data)

    async def mock_send_bytes(data: bytes) -> None:
        sent_bytes.append(data)

    # Infinite LLM stream to simulate user interrupting mid-speech
    class SlowLLM:
        async def stream_completion(
            self,
            messages: list[dict[str, str]],
            cancel_event: asyncio.Event | None = None,
        ) -> AsyncGenerator[str, None]:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return
                yield "talking... "
                await asyncio.sleep(0.05)

    orchestrator = TurnOrchestrator(
        llm_gateway=SlowLLM(),
        tts_engine=MockTTSEngine(),
        stt_engine=MockSTTEngine(),
        send_json=mock_send_json,
        send_bytes=mock_send_bytes,
    )

    await orchestrator.handle_user_message("Start talking")
    await asyncio.sleep(0.02)  # Let it start
    await orchestrator.barge_in()

    types = [msg["type"] for msg in sent_json]
    assert "turn_cancelled" in types
