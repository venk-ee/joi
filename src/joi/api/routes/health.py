"""Health and diagnostic status endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from joi import __version__

router = APIRouter(tags=["observability"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/healthz", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Readiness probe for container orchestration."""
    return HealthResponse(status="ok", version=__version__)
