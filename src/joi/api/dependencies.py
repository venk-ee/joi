"""Dependency injection container for FastAPI routes and WebSocket handlers."""

from functools import lru_cache

from joi.core.config import Settings, get_settings
from joi.domain.protocols import LLMGateway, STTEngine, TTSEngine
from joi.infrastructure.llm.openai_adapter import OpenAILLMAdapter
from joi.infrastructure.speech.kokoro_tts_adapter import KokoroTTSAdapter
from joi.infrastructure.speech.whisper_stt_adapter import WhisperSTTAdapter


@lru_cache(maxsize=1)
def get_tts_engine(settings: Settings | None = None) -> TTSEngine:
    """Provides a singleton Kokoro-82M ONNX TTS engine instance."""
    cfg = settings or get_settings()
    return KokoroTTSAdapter(
        model_path=cfg.kokoro_model_path,
        voices_path=cfg.kokoro_voices_path,
        default_voice=cfg.kokoro_voice,
    )


@lru_cache(maxsize=1)
def get_stt_engine(settings: Settings | None = None) -> STTEngine:
    """Provides a singleton Faster-Whisper INT8 STT engine instance."""
    cfg = settings or get_settings()
    return WhisperSTTAdapter(
        model_size=cfg.whisper_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )


@lru_cache(maxsize=1)
def get_llm_gateway(settings: Settings | None = None) -> LLMGateway:
    """Provides a singleton OpenAI-compatible streaming LLM gateway."""
    cfg = settings or get_settings()
    return OpenAILLMAdapter(
        api_base=cfg.openai_api_base,
        api_key=cfg.openai_api_key,
        model_name=cfg.model_name,
    )
