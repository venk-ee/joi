"""Full-duplex WebSocket hub for real-time control events and binary audio streaming."""

import json
import logging
import struct

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from joi.api.dependencies import get_llm_gateway, get_stt_engine, get_tts_engine
from joi.domain.events import ClientEventType
from joi.domain.protocols import LLMGateway, STTEngine, TTSEngine
from joi.services.orchestrator import TurnOrchestrator

logger = logging.getLogger("joi.websocket")
router = APIRouter(tags=["transport"])

UPLINK_HEADER_FORMAT = "<II"
UPLINK_HEADER_SIZE = struct.calcsize(UPLINK_HEADER_FORMAT)
MESSAGE_TYPE_VOICE_UPLINK = 0x01


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    llm: LLMGateway = Depends(get_llm_gateway),
    tts: TTSEngine = Depends(get_tts_engine),
    stt: STTEngine = Depends(get_stt_engine),
) -> None:
    """Primary bidirectional streaming channel for desktop overlays and mobile AR clients."""
    await websocket.accept()
    logger.info("Client connected to /ws from %s", websocket.client)

    orchestrator = TurnOrchestrator(
        llm_gateway=llm,
        tts_engine=tts,
        stt_engine=stt,
        send_json=websocket.send_json,
        send_bytes=websocket.send_bytes,
    )

    try:
        while True:
            message = await websocket.receive()

            # 1. Binary Frame: Speech uplink from microphone
            if "bytes" in message and message["bytes"] is not None:
                raw_bytes: bytes = message["bytes"]
                if len(raw_bytes) >= UPLINK_HEADER_SIZE:
                    msg_type, sample_rate = struct.unpack_from(UPLINK_HEADER_FORMAT, raw_bytes, 0)
                    if msg_type == MESSAGE_TYPE_VOICE_UPLINK:
                        pcm_payload = raw_bytes[UPLINK_HEADER_SIZE:]
                        logger.debug(
                            "Received speech uplink (%d bytes, %d Hz)",
                            len(pcm_payload),
                            sample_rate,
                        )
                        await orchestrator.handle_user_audio(pcm_payload, sample_rate)

            # 2. Text Frame: JSON control message
            elif "text" in message and message["text"] is not None:
                raw_text: str = message["text"]
                try:
                    payload = json.loads(raw_text)
                    event_type = payload.get("type")

                    if event_type == ClientEventType.USER_MESSAGE:
                        user_text = payload.get("text", "")
                        if user_text:
                            logger.info("Received user_message: '%s'", user_text)
                            await orchestrator.handle_user_message(user_text)

                    elif event_type == ClientEventType.BARGE_IN:
                        logger.info("Received client barge_in")
                        await orchestrator.barge_in()

                    elif event_type == ClientEventType.SPATIAL_TELEMETRY:
                        logger.debug("Received spatial telemetry: %s", payload.get("data"))

                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received over WebSocket: %s", raw_text)

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws")
        await orchestrator.barge_in()
    except Exception as exc:
        logger.error("Unhandled WebSocket error: %s", exc)
        await orchestrator.barge_in()
