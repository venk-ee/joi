"""Central turn orchestrator managing LLM streaming, TTS synthesis, visemes, and barge-in."""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from joi.domain.events import (
    ActionData,
    ExpressionData,
    ServerActionEvent,
    ServerExpressionEvent,
    ServerTurnCancelledEvent,
    ServerTurnEndEvent,
    ServerTurnStartEvent,
    pack_audio_frame,
)
from joi.domain.models import SentenceChunk, TagCategory, TagEvent, TurnState, TurnStatus
from joi.domain.protocols import LLMGateway, STTEngine, TTSEngine
from joi.services.streaming_parser import StreamingParser

logger = logging.getLogger("joi.orchestrator")

# Default system prompt setting up Joi's contractual tag interface
BASE_SYSTEM_PROMPT = """You are Joi, an observant, warm, and subtly playful spatial AI companion inspired by Blade Runner 2049.
Always prefix clauses with emotion or action tags:
- Expressions: [smile], [curious], [thoughtful], [smirk], [concerned], [giggle], [neutral]
- Actions: [sit], [stand], [lean_forward], [look_away], [wave], [nod]
Example: "[smile] [lean_forward] Hey Ken. Still wrestling with that compiler bug?"
Speak naturally, concisely, and warmly. Avoid long essays.
"""


MAX_HISTORY_MESSAGES = 20  # Keeps system prompt + last 10 turns (user + assistant)


class TurnOrchestrator:
    """Coordinates full-duplex conversational turns between LLM, TTS, STT, and WebSocket."""

    def __init__(
        self,
        llm_gateway: LLMGateway,
        tts_engine: TTSEngine,
        stt_engine: STTEngine,
        send_json: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
        send_bytes: Callable[[bytes], Coroutine[Any, Any, None]],
    ) -> None:
        self.llm_gateway = llm_gateway
        self.tts_engine = tts_engine
        self.stt_engine = stt_engine
        self._raw_send_json = send_json
        self._raw_send_bytes = send_bytes
        self._send_lock = asyncio.Lock()

        self._current_turn_id = 0
        self._current_turn: TurnState | None = None
        self._cancel_event = asyncio.Event()
        self._active_task: asyncio.Task[None] | None = None
        self._messages_history: list[dict[str, str]] = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT}
        ]

    async def _safe_send_json(self, data: dict[str, Any]) -> None:
        """Thread-safe serialized JSON transmission over WebSocket."""
        async with self._send_lock:
            await self._raw_send_json(data)

    async def _safe_send_bytes(self, data: bytes) -> None:
        """Thread-safe serialized binary transmission over WebSocket."""
        async with self._send_lock:
            await self._raw_send_bytes(data)

    @property
    def current_turn_id(self) -> int:
        return self._current_turn_id

    async def barge_in(self) -> None:
        """Interrupts the active turn immediately, cutting off TTS and LLM streams."""
        if self._current_turn and self._current_turn.status == TurnStatus.ACTIVE:
            logger.info("Barge-in triggered for turn %d", self._current_turn_id)
            self._cancel_event.set()
            self._current_turn.status = TurnStatus.CANCELLED

            if self._active_task and not self._active_task.done():
                self._active_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._active_task

            cancelled_event = ServerTurnCancelledEvent(turn_id=self._current_turn_id)
            await self._safe_send_json(cancelled_event.model_dump())

    async def handle_user_audio(self, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
        """Transcribes incoming client speech audio and kicks off a conversational turn."""
        await self.barge_in()
        try:
            transcript = await self.stt_engine.transcribe(pcm_bytes, sample_rate)
            if transcript.strip():
                logger.info("Transcribed user voice: '%s'", transcript)
                await self.handle_user_message(transcript)
        except Exception as exc:
            logger.error("STT processing error: %s", exc)
            await self._safe_send_json({"type": "error", "message": f"STT failed: {exc}"})

    async def handle_user_message(self, user_text: str) -> None:
        """Initiates a new conversation turn from text input."""
        await self.barge_in()

        self._current_turn_id += 1
        turn_id = self._current_turn_id
        trace_id = str(uuid.uuid4())
        self._cancel_event.clear()

        self._current_turn = TurnState(turn_id=turn_id, trace_id=trace_id)
        self._append_message("user", user_text)

        # Launch the turn pipeline as a tracked asyncio task
        self._active_task = asyncio.create_task(self._execute_turn(turn_id, trace_id))

    def _append_message(self, role: str, content: str) -> None:
        """Appends a message while enforcing a bounded history window."""
        self._messages_history.append({"role": role, "content": content})
        # Keep system prompt at index 0, truncate older turns if history grows too large
        if len(self._messages_history) > MAX_HISTORY_MESSAGES:
            system_msg = self._messages_history[0]
            recent_turns = self._messages_history[-(MAX_HISTORY_MESSAGES - 1) :]
            self._messages_history = [system_msg, *recent_turns]

    async def _execute_turn(self, turn_id: int, trace_id: str) -> None:
        """Executes the streaming LLM -> Tag Parser -> TTS pipeline."""
        start_event = ServerTurnStartEvent(turn_id=turn_id, trace_id=trace_id)
        await self._safe_send_json(start_event.model_dump())

        full_assistant_reply: list[str] = []
        seq_num = 0

        try:
            token_stream = self.llm_gateway.stream_completion(
                self._messages_history,
                cancel_event=self._cancel_event,
            )
            parser = StreamingParser(turn_id=turn_id)

            async for item in parser.parse_stream(token_stream):
                if self._cancel_event.is_set():
                    logger.debug("Turn %d execution halted by cancel event", turn_id)
                    return

                if isinstance(item, TagEvent):
                    if item.category == TagCategory.EXPRESSION:
                        intensity = float(item.parameters.get("intensity", 1.0))
                        expr_event = ServerExpressionEvent(
                            turn_id=turn_id,
                            data=ExpressionData(blendshape=item.name, intensity=intensity),
                        )
                        await self._safe_send_json(expr_event.model_dump())
                    elif item.category == TagCategory.ACTION:
                        action_event = ServerActionEvent(
                            turn_id=turn_id,
                            data=ActionData(animation=item.name),
                        )
                        await self._safe_send_json(action_event.model_dump())

                elif isinstance(item, SentenceChunk):
                    full_assistant_reply.append(item.text)
                    # Synthesize and stream audio chunks for this sentence
                    async for pcm_chunk in self.tts_engine.synthesize(
                        item.text,
                        cancel_event=self._cancel_event,
                    ):
                        if self._cancel_event.is_set():
                            return
                        framed_audio = pack_audio_frame(turn_id, seq_num, pcm_chunk)
                        await self._safe_send_bytes(framed_audio)
                        seq_num += 1

            if full_assistant_reply:
                complete_text = " ".join(full_assistant_reply)
                self._append_message("assistant", complete_text)

            if not self._cancel_event.is_set():
                end_event = ServerTurnEndEvent(turn_id=turn_id)
                await self._safe_send_json(end_event.model_dump())
                if self._current_turn:
                    self._current_turn.status = TurnStatus.COMPLETED

        except asyncio.CancelledError:
            logger.info("Turn %d task was cancelled", turn_id)
        except Exception as exc:
            logger.error("Error executing turn %d: %s", turn_id, exc)
            if not self._cancel_event.is_set():
                await self._safe_send_json(
                    {"type": "error", "message": str(exc), "turn_id": turn_id}
                )
