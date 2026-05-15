# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests + metadata files required by build backends
COPY graphiti/pyproject.toml graphiti/uv.lock graphiti/README.md graphiti/
COPY mem0/pyproject.toml mem0/poetry.lock mem0/README.md mem0/LICENSE mem0/
COPY neuralscape-service/pyproject.toml neuralscape-service/uv.lock neuralscape-service/

# Copy source packages needed for editable installs
COPY graphiti/graphiti_core/ graphiti/graphiti_core/
COPY mem0/mem0/ mem0/mem0/
COPY neuralscape-service/*.py neuralscape-service/
COPY neuralscape-service/extensions/ neuralscape-service/extensions/

# Install all dependencies using uv from the neuralscape-service project
RUN cd neuralscape-service && uv sync --frozen --no-dev

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/neuralscape-service/.venv /app/neuralscape-service/.venv

# Copy source code
COPY graphiti/graphiti_core/ graphiti/graphiti_core/
COPY graphiti/pyproject.toml graphiti/
COPY mem0/mem0/ mem0/mem0/
COPY mem0/pyproject.toml mem0/
COPY neuralscape-service/*.py neuralscape-service/
COPY neuralscape-service/extensions/ neuralscape-service/extensions/

# Put venv on PATH
ENV PATH="/app/neuralscape-service/.venv/bin:$PATH"

# Default environment for container networking
ENV NEO4J_URI=neo4j://neo4j:7687 \
    QDRANT_PATH=/data/qdrant

WORKDIR /app/neuralscape-service

# Run as non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /data/qdrant /data/vault && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8199

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8199"]

# ── Stage 3: Test ───────────────────────────────────────────────
# Mirrors runtime but includes dev dependencies (pytest, etc.) and the
# tests/ + scripts/ directories so the same image can run the suite.
# Build with `--target test` or `docker compose --profile test`.
FROM python:3.12-slim AS test

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Reuse the same dependency-resolution copy pattern as the builder.
COPY graphiti/pyproject.toml graphiti/uv.lock graphiti/README.md graphiti/
COPY mem0/pyproject.toml mem0/poetry.lock mem0/README.md mem0/LICENSE mem0/
COPY neuralscape-service/pyproject.toml neuralscape-service/uv.lock neuralscape-service/

# Source packages needed for editable installs.
COPY graphiti/graphiti_core/ graphiti/graphiti_core/
COPY mem0/mem0/ mem0/mem0/
COPY neuralscape-service/*.py neuralscape-service/
COPY neuralscape-service/extensions/ neuralscape-service/extensions/

# Resolve WITH dev deps so pytest, pytest-asyncio, etc. are available.
RUN cd neuralscape-service && uv sync --frozen

# Pull tests + helper scripts into the image so they're available without
# a host bind-mount. Bind-mounts can still be layered on top from compose
# for faster iteration; the baked-in copy guarantees the image is
# reproducibly testable on its own.
COPY neuralscape-service/tests/ neuralscape-service/tests/
COPY neuralscape-service/scripts/ neuralscape-service/scripts/

ENV PATH="/app/neuralscape-service/.venv/bin:$PATH"
ENV NEO4J_URI=neo4j://neo4j:7687 \
    QDRANT_PATH=/data/qdrant \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/neuralscape-service

# Same non-root user pattern as runtime, so file-permission behavior
# during tests mirrors production.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /data/qdrant /data/vault && chown -R appuser:appuser /data /app
USER appuser

# Default to running the full suite minus the integration tests that
# require a live ARQ worker. Compose overrides this via `command:`.
CMD ["pytest", "tests/", "--ignore=tests/test_async_pipeline.py", "-q"]
