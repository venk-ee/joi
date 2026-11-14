"""Pre-fetches Kokoro ONNX model files and Faster-Whisper weights for offline use."""

import asyncio
from pathlib import Path

from joi.core.config import get_settings
from joi.core.logging import setup_logger
from joi.infrastructure.speech.kokoro_tts_adapter import KokoroTTSAdapter
from joi.infrastructure.speech.whisper_stt_adapter import WhisperSTTAdapter

logger = setup_logger("joi.downloader")


async def main() -> None:
    settings = get_settings()
    logger.info("Pre-fetching speech model assets into %s...", Path("models").resolve())

    # 1. Download & warmup Kokoro TTS
    tts = KokoroTTSAdapter(
        model_path=settings.kokoro_model_path,
        voices_path=settings.kokoro_voices_path,
        default_voice=settings.kokoro_voice,
    )
    await tts.ensure_assets()
    logger.info("Kokoro assets ready.")

    # 2. Pre-fetch Faster-Whisper weights
    stt = WhisperSTTAdapter(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    await stt.warmup()
    logger.info("Faster-Whisper weights ready.")
    logger.info("All model assets cached successfully!")


if __name__ == "__main__":
    asyncio.run(main())
