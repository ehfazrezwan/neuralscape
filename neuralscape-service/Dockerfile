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
COPY neuralscape-service/knowledge/ neuralscape-service/knowledge/
COPY neuralscape-service/ingest/ neuralscape-service/ingest/
COPY neuralscape-service/connectors/ neuralscape-service/connectors/
COPY neuralscape-service/okf/ neuralscape-service/okf/
COPY neuralscape-service/integration_tests/ neuralscape-service/integration_tests/

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

# Non-root user created first so COPY --chown can assign ownership while
# files are written, instead of a recursive chown layer that duplicates every
# file into a new image layer. COPY-created implicit parent directories are
# root-owned, so the application directories are pre-created and chowned
# (non-recursively) here.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/neuralscape-service /app/graphiti /app/mem0 \
        /data/qdrant /data/vault /data/ingest /data/exemplars && \
    chown appuser:appuser /app /app/neuralscape-service /app/graphiti /app/mem0 \
        /data /data/qdrant /data/vault /data/ingest /data/exemplars

WORKDIR /app

# Copy the virtual environment from builder
COPY --chown=appuser:appuser --from=builder /app/neuralscape-service/.venv /app/neuralscape-service/.venv

# Copy source code
COPY --chown=appuser:appuser graphiti/graphiti_core/ graphiti/graphiti_core/
COPY --chown=appuser:appuser graphiti/pyproject.toml graphiti/
COPY --chown=appuser:appuser mem0/mem0/ mem0/mem0/
COPY --chown=appuser:appuser mem0/pyproject.toml mem0/
COPY --chown=appuser:appuser neuralscape-service/*.py neuralscape-service/
COPY --chown=appuser:appuser neuralscape-service/memory/ neuralscape-service/memory/
COPY --chown=appuser:appuser neuralscape-service/extensions/ neuralscape-service/extensions/
COPY --chown=appuser:appuser neuralscape-service/adapters/ neuralscape-service/adapters/
COPY --chown=appuser:appuser neuralscape-service/knowledge/ neuralscape-service/knowledge/
COPY --chown=appuser:appuser neuralscape-service/ingest/ neuralscape-service/ingest/
COPY --chown=appuser:appuser neuralscape-service/connectors/ neuralscape-service/connectors/
COPY --chown=appuser:appuser neuralscape-service/okf/ neuralscape-service/okf/
COPY --chown=appuser:appuser neuralscape-service/integration_tests/ neuralscape-service/integration_tests/
# Operator-facing scripts (relocate_vault_to_raw, issue_user_token,
# migrate_graph_groups, bulk_promote_visibility). Production operators
# invoke these in-container; without this copy they had to docker-cp
# them in by hand.
COPY --chown=appuser:appuser neuralscape-service/scripts/ neuralscape-service/scripts/

# Put venv on PATH
ENV PATH="/app/neuralscape-service/.venv/bin:$PATH"

# Default environment for container networking
ENV NEO4J_URI=neo4j://neo4j:7687 \
    QDRANT_PATH=/data/qdrant

# Vendored mem0/Graphiti send anonymous usage telemetry to an external host
# unless disabled. Off by default; opt in by setting either variable at run time.
ENV MEM0_TELEMETRY=false \
    GRAPHITI_TELEMETRY_ENABLED=false

WORKDIR /app/neuralscape-service

# Run as non-root user (created and given ownership at the top of this stage)
USER appuser

EXPOSE 8199

# Configurable uvicorn worker count via UVICORN_WORKERS env var (default 1).
# Multiple workers enable concurrent request handling across CPU cores, but each
# holds its own interpreter/model state. Default 1 preserves single-process behavior.
# The expansion is validated as a positive integer and quoted so a stray/whitespace
# value can't word-split into unintended uvicorn arguments (falls back to 1).
CMD ["sh", "-c", "w=\"${UVICORN_WORKERS:-1}\"; case \"$w\" in ''|*[!0-9]*) w=1;; esac; [ \"$w\" -ge 1 ] 2>/dev/null || w=1; exec uvicorn main:app --host 0.0.0.0 --port 8199 --workers \"$w\""]

# ── Stage 3: Test ───────────────────────────────────────────────
# Mirrors runtime but includes dev dependencies (pytest, etc.) and the
# tests/ + scripts/ directories so the same image can run the suite.
# Build with `--target test` or `docker compose --profile test`.
FROM python:3.12-slim AS test

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Non-root user created first so COPY --chown can assign ownership while
# files are written, instead of a recursive chown layer that duplicates every
# file into a new image layer. COPY-created implicit parent directories are
# root-owned, so the application directories are pre-created and chowned
# (non-recursively) here.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/neuralscape-service /app/graphiti /app/mem0 \
        /data/qdrant /data/vault /data/ingest /data/exemplars && \
    chown appuser:appuser /app /app/neuralscape-service /app/graphiti /app/mem0 \
        /data /data/qdrant /data/vault /data/ingest /data/exemplars

# Install and run as the service user so the venv is created appuser-owned
# (uv cache lands in /home/appuser/.cache).
USER appuser
WORKDIR /app

# Reuse the same dependency-resolution copy pattern as the builder.
COPY --chown=appuser:appuser graphiti/pyproject.toml graphiti/uv.lock graphiti/README.md graphiti/
COPY --chown=appuser:appuser mem0/pyproject.toml mem0/poetry.lock mem0/README.md mem0/LICENSE mem0/
COPY --chown=appuser:appuser neuralscape-service/pyproject.toml neuralscape-service/uv.lock neuralscape-service/

# Source packages needed for editable installs.
COPY --chown=appuser:appuser graphiti/graphiti_core/ graphiti/graphiti_core/
COPY --chown=appuser:appuser mem0/mem0/ mem0/mem0/
COPY --chown=appuser:appuser neuralscape-service/*.py neuralscape-service/
COPY --chown=appuser:appuser neuralscape-service/memory/ neuralscape-service/memory/
COPY --chown=appuser:appuser neuralscape-service/extensions/ neuralscape-service/extensions/
COPY --chown=appuser:appuser neuralscape-service/adapters/ neuralscape-service/adapters/
COPY --chown=appuser:appuser neuralscape-service/knowledge/ neuralscape-service/knowledge/
COPY --chown=appuser:appuser neuralscape-service/ingest/ neuralscape-service/ingest/
COPY --chown=appuser:appuser neuralscape-service/connectors/ neuralscape-service/connectors/
COPY --chown=appuser:appuser neuralscape-service/okf/ neuralscape-service/okf/

# Resolve WITH dev deps so pytest, pytest-asyncio, etc. are available.
# E2: include the code-graph extra so in-container tests can import tree-sitter
RUN cd neuralscape-service && uv sync --frozen --extra code-graph

# Same spaCy model handling as the runtime stage so unit tests that
# import mem0's NLP path see the same dependency graph as production.
COPY --chown=appuser:appuser neuralscape-service/vendor/ neuralscape-service/vendor/
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
COPY --chown=appuser:appuser neuralscape-service/tests/ neuralscape-service/tests/
# Test-stage only (the runtime image does not carry it): tests that assert on
# the Dockerfile itself read it from here.
COPY --chown=appuser:appuser neuralscape-service/Dockerfile neuralscape-service/Dockerfile
COPY --chown=appuser:appuser neuralscape-service/integration_tests/ neuralscape-service/integration_tests/
COPY --chown=appuser:appuser neuralscape-service/scripts/ neuralscape-service/scripts/

ENV PATH="/app/neuralscape-service/.venv/bin:$PATH"
ENV NEO4J_URI=neo4j://neo4j:7687 \
    QDRANT_PATH=/data/qdrant \
    PYTHONDONTWRITEBYTECODE=1 \
    MEM0_TELEMETRY=false \
    GRAPHITI_TELEMETRY_ENABLED=false

WORKDIR /app/neuralscape-service

# Already running as appuser (set before the dependency install), so
# file-permission behavior during tests mirrors production.

# Default to running the full suite minus the integration tests that
# require a live ARQ worker. Compose overrides this via `command:`.
CMD ["pytest", "tests/", "--ignore=tests/test_async_pipeline.py", "-q"]
