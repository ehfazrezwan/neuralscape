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

# Install all dependencies using uv from the neuralscape-service project
RUN cd neuralscape-service && uv sync --frozen --no-dev

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/neuralscape-service/.venv /app/neuralscape-service/.venv

# Copy source code
COPY graphiti/graphiti_core/ graphiti/graphiti_core/
COPY graphiti/pyproject.toml graphiti/
COPY mem0/mem0/ mem0/mem0/
COPY mem0/pyproject.toml mem0/
COPY neuralscape-service/*.py neuralscape-service/

# Put venv on PATH
ENV PATH="/app/neuralscape-service/.venv/bin:$PATH"

# Default environment for container networking
ENV NEO4J_URI=neo4j://neo4j:7687 \
    QDRANT_PATH=/data/qdrant

WORKDIR /app/neuralscape-service

EXPOSE 8199

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8199"]
