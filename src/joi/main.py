"""Main entry point and application factory for the Joi spatial AI companion."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from joi import __version__
from joi.api.dependencies import get_stt_engine, get_tts_engine
from joi.api.routes import health, websocket
from joi.core.config import Settings, get_settings
from joi.core.logging import setup_logger

logger = setup_logger("joi.server")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and teardown lifecycle."""
    settings = get_settings()
    logger.info("Starting Joi Spatial AI Companion v%s...", __version__)
    logger.info("BYOK Base: %s | Model: %s", settings.openai_api_base, settings.model_name)

    # Warm up CPU speech models during boot to eliminate first-interaction latency
    try:
        tts = get_tts_engine(settings)
        await tts.warmup()
    except Exception as exc:
        logger.warning("TTS warmup encountered warning (will retry on first turn): %s", exc)

    try:
        stt = get_stt_engine(settings)
        await stt.warmup()
    except Exception as exc:
        logger.warning("STT warmup encountered warning (will retry on first turn): %s", exc)

    logger.info("Joi Core Engine initialization complete. Ready for connections.")
    yield
    logger.info("Shutting down Joi Core Engine...")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory constructing the FastAPI server with dependency wiring."""
    app = FastAPI(
        title="Joi Spatial AI Companion Core Engine",
        version=__version__,
        description="Headless multimodal, spatial voice engine inspired by Blade Runner 2049.",
        lifespan=lifespan,
    )

    # Enable CORS for local development and LAN / WebXR access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API and WebSocket routes
    app.include_router(health.router)
    app.include_router(websocket.router)

    # Serve static 3D actuator frontend if present
    frontend_dir = Path("frontend")
    if frontend_dir.exists() and (frontend_dir / "index.html").exists():
        logger.info("Serving static frontend from %s", frontend_dir.resolve())
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


def run() -> None:
    """CLI launcher for uvicorn server."""
    settings = get_settings()
    uvicorn.run(
        "joi.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
