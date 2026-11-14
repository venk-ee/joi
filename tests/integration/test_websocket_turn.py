"""Integration test for full-duplex WebSocket interaction using TestClient."""

import asyncio
import json
from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from joi.api.dependencies import get_llm_gateway, get_stt_engine, get_tts_engine
from joi.domain.events import unpack_audio_frame
from joi.main import create_app


class StubLLM:
    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        for token in ["[smile] ", "Hey Ken. ", "[wave] ", "Nice to see you!"]:
            if cancel_event is not None and cancel_event.is_set():
                return
            yield token


class StubTTS:
    async def warmup(self) -> None:
        pass

    async def synthesize(
        self,
        text: str,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[bytes, None]:
        if cancel_event is not None and cancel_event.is_set():
            return
        yield b"\x00\x01\x02\x03" * 20


class StubSTT:
    async def warmup(self) -> None:
        pass

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        return "Hello Joi"


@pytest.fixture
def test_app() -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_llm_gateway] = lambda: StubLLM()
    app.dependency_overrides[get_tts_engine] = lambda: StubTTS()
    app.dependency_overrides[get_stt_engine] = lambda: StubSTT()
    return app


def test_healthz_endpoint(test_app: FastAPI) -> None:
    client = TestClient(test_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_websocket_text_turn(test_app: FastAPI) -> None:
    client = TestClient(test_app)
    with client.websocket_connect("/ws") as ws:
        # Send user text turn
        ws.send_json({"type": "user_message", "text": "Hey Joi"})

        received_events: list[dict[str, object]] = []
        received_bytes: list[bytes] = []

        # Read frames until turn_end
        while True:
            msg = ws.receive()
            if "text" in msg and msg["text"]:
                parsed = json.loads(msg["text"])
                received_events.append(parsed)
                if parsed.get("type") == "turn_end":
                    break
            elif "bytes" in msg and msg["bytes"]:
                received_bytes.append(msg["bytes"])

        types = [e.get("type") for e in received_events]
        assert "turn_start" in types
        assert "expression" in types
        assert "action" in types
        assert "turn_end" in types

        # Check binary audio frames
        assert len(received_bytes) == 2
        turn_id, seq_num, pcm = unpack_audio_frame(received_bytes[0])
        assert turn_id == 1
        assert seq_num == 0
        assert len(pcm) == 80
