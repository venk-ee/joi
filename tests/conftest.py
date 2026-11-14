"""Pytest configuration and shared test fixtures."""

import pytest

from joi.core.config import Settings


@pytest.fixture
def mock_settings() -> Settings:
    """Provides test-specific configuration settings."""
    return Settings(
        openai_api_base="https://mock.api/v1",
        openai_api_key="mock-key",
        model_name="mock-model",
        log_level="DEBUG",
        debug=True,
    )
