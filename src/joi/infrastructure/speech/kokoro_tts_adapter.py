"""Kokoro-82M ONNX Text-to-Speech adapter running on CPU."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import numpy as np
from kokoro_onnx import Kokoro

from joi.core.exceptions import AudioSynthesisError

logger = logging.getLogger("joi.tts")

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)


class KokoroTTSAdapter:
    """CPU-offloaded TTS adapter implementing the TTSEngine protocol."""

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        default_voice: str = "af_heart",
    ) -> None:
        self.model_path = model_path
        self.voices_path = voices_path
        self.default_voice = default_voice
        self._kokoro: Kokoro | None = None
        self._lock = asyncio.Lock()

    async def _download_file(self, client: httpx.AsyncClient, url: str, target_path: Path) -> None:
        """Streams a download chunk-by-chunk to a temporary file before atomic rename."""
        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
            temp_path.replace(target_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def ensure_assets(self) -> None:
        """Downloads Kokoro weights and voice embeddings if they are not already cached."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(follow_redirects=True, timeout=180.0) as client:
            if not self.model_path.exists():
                logger.info("Downloading Kokoro ONNX model weights to %s...", self.model_path)
                await self._download_file(client, KOKORO_MODEL_URL, self.model_path)
                logger.info("Kokoro ONNX model weights downloaded successfully.")

            if not self.voices_path.exists():
                logger.info("Downloading Kokoro voice embeddings to %s...", self.voices_path)
                await self._download_file(client, KOKORO_VOICES_URL, self.voices_path)
                logger.info("Kokoro voice embeddings downloaded successfully.")

    async def warmup(self) -> None:
        """Initializes Kokoro engine and runs a dummy synthesis to eliminate first-turn latency."""
        async with self._lock:
            if self._kokoro is None:
                await self.ensure_assets()
                logger.info("Loading Kokoro ONNX model into CPU session...")
                self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))

            logger.info("Warming up Kokoro ONNX engine...")
            # Run short warmup synthesis in a thread pool to avoid blocking the event loop
            _ = await asyncio.to_thread(
                self._kokoro.create, "Hello.", voice=self.default_voice, speed=1.0
            )
            logger.info("Kokoro ONNX engine warm and ready.")

    async def synthesize(
        self,
        text: str,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesizes text into 24kHz 16-bit linear PCM audio chunks.

        Periodically checks cancel_event to support instant barge-in cancellation.
        """
        if not text.strip():
            return

        if self._kokoro is None:
            await self.warmup()
            assert self._kokoro is not None

        try:
            stream = self._kokoro.create_stream(
                text=text,
                voice=self.default_voice,
                speed=1.0,
                lang="en-us",
            )

            async for samples, _sample_rate in stream:
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("TTS synthesis cancelled by barge-in event.")
                    return

                # Convert float32 [-1.0, 1.0] to int16 linear PCM
                int16_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                pcm_bytes = int16_samples.tobytes()
                yield pcm_bytes

        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                return
            logger.error("TTS synthesis error for text '%s': %s", text, exc)
            raise AudioSynthesisError(f"Kokoro synthesis failed: {exc}") from exc
