"""Domain exception hierarchy for the Joi core engine."""


class JoiError(Exception):
    """Base exception for all Joi engine domain errors."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AudioSynthesisError(JoiError):
    """Raised when text-to-speech synthesis fails or model cannot be loaded."""


class SpeechRecognitionError(JoiError):
    """Raised when speech-to-text transcription fails."""


class LLMStreamError(JoiError):
    """Raised when the LLM SSE streaming connection encounters an unrecoverable failure."""


class MemoryStorageError(JoiError):
    """Raised when SQLite FTS5 episodic memory query or persistence fails."""


class InvalidEventError(JoiError):
    """Raised when incoming client WebSocket frames fail validation."""
