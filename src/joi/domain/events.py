"""Strict Pydantic V2 wire events and binary frame helpers for the WebSocket protocol."""

import struct
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# Binary Audio Header: 4 bytes turn_id (uint32-LE) + 4 bytes seq_num (uint32-LE) = 8 bytes
BINARY_HEADER_FORMAT = "<II"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)


def pack_audio_frame(turn_id: int, seq_num: int, pcm_bytes: bytes) -> bytes:
    """Packs an 8-byte header onto raw PCM audio bytes for deterministic turn routing."""
    header = struct.pack(BINARY_HEADER_FORMAT, turn_id, seq_num)
    return header + pcm_bytes


def unpack_audio_frame(raw_data: bytes) -> tuple[int, int, bytes]:
    """Unpacks an 8-byte framed audio packet into (turn_id, seq_num, pcm_bytes)."""
    if len(raw_data) < BINARY_HEADER_SIZE:
        raise ValueError(
            f"Binary frame too short: {len(raw_data)} bytes, expected at least {BINARY_HEADER_SIZE}"
        )
    turn_id, seq_num = struct.unpack_from(BINARY_HEADER_FORMAT, raw_data, 0)
    pcm_bytes = raw_data[BINARY_HEADER_SIZE:]
    return turn_id, seq_num, pcm_bytes


# --- Server -> Client JSON Control Events ---


class ServerEventType(StrEnum):
    TURN_START = "turn_start"
    EXPRESSION = "expression"
    ACTION = "action"
    TURN_END = "turn_end"
    TURN_CANCELLED = "turn_cancelled"


class ExpressionData(BaseModel):
    blendshape: str
    intensity: float = Field(ge=0.0, le=1.0, default=1.0)
    duration_ms: int = Field(default=1000)


class ActionData(BaseModel):
    animation: str
    fade_ms: int = Field(default=400)


class ServerTurnStartEvent(BaseModel):
    type: ServerEventType = ServerEventType.TURN_START
    turn_id: int
    trace_id: str


class ServerExpressionEvent(BaseModel):
    type: ServerEventType = ServerEventType.EXPRESSION
    turn_id: int
    data: ExpressionData


class ServerActionEvent(BaseModel):
    type: ServerEventType = ServerEventType.ACTION
    turn_id: int
    data: ActionData


class ServerTurnEndEvent(BaseModel):
    type: ServerEventType = ServerEventType.TURN_END
    turn_id: int


class ServerTurnCancelledEvent(BaseModel):
    type: ServerEventType = ServerEventType.TURN_CANCELLED
    turn_id: int


# --- Client -> Server JSON Control Events ---


class ClientEventType(StrEnum):
    USER_MESSAGE = "user_message"
    BARGE_IN = "barge_in"
    SPATIAL_TELEMETRY = "spatial_telemetry"


class ClientMessageEvent(BaseModel):
    type: ClientEventType = ClientEventType.USER_MESSAGE
    text: str


class ClientBargeInEvent(BaseModel):
    type: ClientEventType = ClientEventType.BARGE_IN


class SpatialData(BaseModel):
    plane_height: float | None = None
    ambient_light: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ClientSpatialTelemetryEvent(BaseModel):
    type: ClientEventType = ClientEventType.SPATIAL_TELEMETRY
    data: SpatialData
