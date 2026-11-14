"""Headless interactive CLI client for benchmarking latency, events, and audio streaming."""

import asyncio
import json
import logging
import time

import websockets

from joi.domain.events import unpack_audio_frame

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mock_client")


async def run_client(server_url: str = "ws://localhost:8000/ws") -> None:
    logger.info("Connecting to Joi Core Engine at %s...", server_url)
    async with websockets.connect(server_url) as ws:
        logger.info("Connected! Type a message to speak to Joi, or type 'quit' to exit.")
        logger.info("Type '[barge]' mid-response to simulate user interruption.")

        async def receive_loop() -> None:
            first_event_time: float | None = None
            first_audio_time: float | None = None
            start_time: float = time.time()
            total_audio_bytes = 0

            try:
                while True:
                    msg = await ws.recv()
                    now = time.time()

                    if isinstance(msg, str):
                        data = json.loads(msg)
                        event_type = data.get("type")

                        if first_event_time is None:
                            first_event_time = now
                            logger.info(
                                "⚡ [First Event] %s in %.1f ms",
                                event_type,
                                (first_event_time - start_time) * 1000,
                            )

                        if event_type == "turn_start":
                            logger.info("▶️ Turn %s started", data.get("turn_id"))
                        elif event_type == "expression":
                            logger.info(
                                "😊 Expression: %s (intensity: %.1f)",
                                data["data"]["blendshape"],
                                data["data"]["intensity"],
                            )
                        elif event_type == "action":
                            logger.info("🎬 Action: %s", data["data"]["animation"])
                        elif event_type == "turn_end":
                            logger.info(
                                "⏹️ Turn %s ended (Total Audio: %d KB)",
                                data.get("turn_id"),
                                total_audio_bytes // 1024,
                            )
                        elif event_type == "turn_cancelled":
                            logger.info("🚫 Turn %s cancelled by barge-in", data.get("turn_id"))

                    elif isinstance(msg, bytes):
                        turn_id, seq_num, pcm_payload = unpack_audio_frame(msg)
                        total_audio_bytes += len(pcm_payload)

                        if first_audio_time is None:
                            first_audio_time = now
                            logger.info(
                                "🔊 [First Audio Chunk] seq=%d, %d bytes in %.1f ms",
                                seq_num,
                                len(pcm_payload),
                                (first_audio_time - start_time) * 1000,
                            )

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server.")

        recv_task = asyncio.create_task(receive_loop())

        loop = asyncio.get_running_loop()
        while True:
            user_input = await loop.run_in_executor(None, input, "\nYou > ")
            if not user_input.strip():
                continue
            if user_input.strip().lower() == "quit":
                break
            if user_input.strip().lower() == "[barge]":
                await ws.send(json.dumps({"type": "barge_in"}))
                continue

            await ws.send(json.dumps({"type": "user_message", "text": user_input}))

        recv_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        logger.info("\nExiting.")
