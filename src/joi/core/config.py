"""Application configuration using Pydantic Settings (12-Factor compliant)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for Joi Core Engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- BYOK LLM Configuration ---
    openai_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenAI-compatible LLM inference",
    )
    openai_api_key: str = Field(
        default="",
        description="API key for LLM provider (BYOK)",
    )
    model_name: str = Field(
        default="cognitivecomputations/dolphin-mixtral-8x7b",
        description="Target model identifier",
    )

    # --- Speech Configuration (CPU-Offloaded) ---
    kokoro_model_path: Path = Field(
        default=Path("models/kokoro-v1.0.onnx"),
        description="Path to Kokoro ONNX model weights",
    )
    kokoro_voices_path: Path = Field(
        default=Path("models/voices-v1.0.bin"),
        description="Path to Kokoro voice embeddings",
    )
    kokoro_voice: str = Field(
        default="af_heart",
        description="Default Kokoro voice identifier",
    )

    whisper_model_size: str = Field(
        default="base.en",
        description="Faster-Whisper model size (tiny.en, base.en, small.en)",
    )
    whisper_device: str = Field(
        default="cpu",
        description="Compute device for Whisper (CPU keeps GPU free)",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="Quantization for Whisper on CPU",
    )

    # --- Server & Transport ---
    host: str = Field(default="0.0.0.0", description="Host address for FastAPI")
    port: int = Field(default=8000, description="Listening port")
    log_level: str = Field(default="INFO", description="Log verbosity level")
    debug: bool = Field(default=False, description="Debug mode")

    # --- Memory Persistence ---
    database_url: str = Field(
        default="sqlite:///data/joi_memory.db",
        description="SQLite database path for episodic FTS5 memory",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton cached provider for application settings."""
    return Settings()
