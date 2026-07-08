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
COPY neuralscape-service/memory/ neuralscape-service/memory/
COPY neuralscape-service/extensions/ neuralscape-service/extensions/
COPY neuralscape-service/adapters/ neuralscape-service/adapters/
COPY neuralscape-service/ingest/ neuralscape-service/ingest/
COPY neuralscape-service/connectors/ neuralscape-service/connectors/
COPY neuralscape-service/okf/ neuralscape-service/okf/

# Install all dependencies using uv from the neuralscape-service project
# E2: include the code-graph extra (tree-sitter + graphifyy) in the builder
RUN cd neuralscape-service && uv sync --frozen --no-dev --extra code-graph

# Download the small English spaCy model so mem0's keyword-extraction
# layer finds it at import time. Without this the worker logs a
# "Failed to load spaCy lemma model" warning on every startup and
# falls back to no lemmatization, hurting search quality.
# A pre-downloaded wheel in neuralscape-service/vendor/ takes priority —
# needed on hosts whose TLS is intercepted (e.g. Cisco Umbrella), where
# the build container can't validate GitHub's release CDN certificate.
# (COPY copies the directory's contents into the destination, so the wheel
# lands at vendor/en_core_web_sm-*.whl, not vendor/vendor/.)
COPY neuralscape-service/vendor/ neuralscape-service/vendor/
RUN cd neuralscape-service && \
    set -- vendor/en_core_web_sm-*.whl; \
    if [ -f "$1" ]; then \
        [ "$#" -eq 1 ] || { echo "expected exactly one en_core_web_sm wheel in vendor/, found $#" >&2; exit 1; }; \
        uv pip install --python .venv/bin/python "$1"; \
    else \
        .venv/bin/python -m spacy download en_core_web_sm; \
    fi

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
COPY neuralscape-service/memory/ neuralscape-service/memory/
COPY neuralscape-service/extensions/ neuralscape-service/extensions/
COPY neuralscape-service/adapters/ neuralscape-service/adapters/
COPY neuralscape-service/ingest/ neuralscape-service/ingest/
COPY neuralscape-service/connectors/ neuralscape-service/connectors/
COPY neuralscape-service/okf/ neuralscape-service/okf/
# Operator-facing scripts (relocate_vault_to_raw, issue_user_token,
# migrate_graph_groups, bulk_promote_visibility). Production operators
# invoke these in-container; without this copy they had to docker-cp
# them in by hand.
COPY neuralscape-service/scripts/ neuralscape-service/scripts/

# Put venv on PATH
ENV PATH="/app/neuralscape-service/.venv/bin:$PATH"

# Default environment for container networking
ENV NEO4J_URI=neo4j://neo4j:7687 \
    QDRANT_PATH=/data/qdrant

WORKDIR /app/neuralscape-service

# Run as non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /data/qdrant /data/vault /data/ingest /data/exemplars && chown -R appuser:appuser /data /app
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
COPY neuralscape-service/memory/ neuralscape-service/memory/
COPY neuralscape-service/extensions/ neuralscape-service/extensions/
COPY neuralscape-service/adapters/ neuralscape-service/adapters/
COPY neuralscape-service/ingest/ neuralscape-service/ingest/
COPY neuralscape-service/connectors/ neuralscape-service/connectors/
COPY neuralscape-service/okf/ neuralscape-service/okf/

# Resolve WITH dev deps so pytest, pytest-asyncio, etc. are available.
# E2: include the code-graph extra so in-container tests can import tree-sitter
RUN cd neuralscape-service && uv sync --frozen --extra code-graph

# Same spaCy model handling as the runtime stage so unit tests that
# import mem0's NLP path see the same dependency graph as production.
COPY neuralscape-service/vendor/ neuralscape-service/vendor/
RUN cd neuralscape-service && \
    set -- vendor/en_core_web_sm-*.whl; \
    if [ -f "$1" ]; then \
        [ "$#" -eq 1 ] || { echo "expected exactly one en_core_web_sm wheel in vendor/, found $#" >&2; exit 1; }; \
        uv pip install --python .venv/bin/python "$1"; \
    else \
        .venv/bin/python -m spacy download en_core_web_sm; \
    fi

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
    mkdir -p /data/qdrant /data/vault /data/ingest /data/exemplars && chown -R appuser:appuser /data /app
USER appuser

# Default to running the full suite minus the integration tests that
# require a live ARQ worker. Compose overrides this via `command:`.
CMD ["pytest", "tests/", "--ignore=tests/test_async_pipeline.py", "-q"]
