"""Faster-Whisper Speech-to-Text adapter running quantized on CPU."""

import asyncio
import logging

import numpy as np
from faster_whisper import WhisperModel

from joi.core.exceptions import SpeechRecognitionError

logger = logging.getLogger("joi.stt")


class WhisperSTTAdapter:
    """CPU-offloaded STT adapter implementing the STTEngine protocol."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None
        self._lock = asyncio.Lock()

    async def warmup(self) -> None:
        """Initializes Faster-Whisper on CPU in a background thread."""
        async with self._lock:
            if self._model is None:
                logger.info(
                    "Loading Faster-Whisper (%s, %s, %s)...",
                    self.model_size,
                    self.device,
                    self.compute_type,
                )
                self._model = await asyncio.to_thread(
                    WhisperModel,
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info("Faster-Whisper initialized and ready.")

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribes raw 16-bit PCM audio bytes to text without blocking the async event loop."""
        if not pcm_bytes:
            return ""

        if self._model is None:
            await self.warmup()
            assert self._model is not None

        try:
            # Convert int16 PCM bytes to float32 array normalized to [-1.0, 1.0]
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample to 16,000 Hz if client audio captured at 44.1kHz or 48kHz
            if sample_rate != 16000 and len(audio_array) > 0:
                target_length = int(len(audio_array) * 16000 / sample_rate)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), target_length, endpoint=False),
                    np.arange(len(audio_array)),
                    audio_array,
                ).astype(np.float32)

            def _run_transcribe() -> str:
                assert self._model is not None
                segments, _ = self._model.transcribe(
                    audio_array,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                )
                return " ".join(seg.text.strip() for seg in segments).strip()

            transcript = await asyncio.to_thread(_run_transcribe)
            return transcript

        except Exception as exc:
            logger.error("STT transcription error: %s", exc)
            raise SpeechRecognitionError(f"Faster-Whisper transcription failed: {exc}") from exc
