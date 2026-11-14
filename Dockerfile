# Multi-stage production build using uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first for Docker layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

# Final slim runtime image
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Install system runtime dependencies (e.g. libgomp for onnx/ctranslate2, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code and frontend
COPY src ./src
COPY frontend ./frontend

# Create persistent mount points
RUN mkdir -p /app/models /app/data

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["python", "-m", "joi.main"]
