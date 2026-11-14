"""Domain models and value objects for the Joi spatial companion."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class TurnStatus(StrEnum):
    """Lifecycle status of a conversation turn."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TagCategory(StrEnum):
    """Classification of inline LLM tags."""

    EXPRESSION = "expression"
    ACTION = "action"


@dataclass(frozen=True)
class TagEvent:
    """An inline tag parsed from the LLM token stream."""

    name: str
    category: TagCategory
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SentenceChunk:
    """A segment of text bounded by punctuation ready for speech synthesis."""

    text: str
    sentence_idx: int
    turn_id: int


@dataclass(frozen=True)
class AudioChunk:
    """A binary PCM audio frame dispatched to the client."""

    turn_id: int
    seq_num: int
    pcm_bytes: bytes
    sample_rate: int = 24000
    channels: int = 1


@dataclass
class TurnState:
    """Runtime tracking state for an in-flight conversation turn."""

    turn_id: int
    trace_id: str
    status: TurnStatus = TurnStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    sentences_emitted: int = 0
    audio_chunks_emitted: int = 0


@dataclass(frozen=True)
class MemoryFact:
    """A long-term episodic fact stored in SQLite FTS5."""

    content: str
    category: str  # 'preference', 'project', 'habit', 'emotion'
    importance: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
