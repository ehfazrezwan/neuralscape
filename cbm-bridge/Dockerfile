# CBM Bridge Service Dockerfile
#
# Downloads and runs a thin FastAPI REST shim over the CBM binary (MIT-licensed).
# The bridge exposes structured JSON tools only (no raw query_graph Cypher).
# Secrecy-clean: no org identifiers.
#
# CBM binary is downloaded from GitHub releases (DeusData/codebase-memory-mcp,
# MIT license, v0.9.0). Source: https://github.com/DeusData/codebase-memory-mcp

FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libsqlite3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download CBM binary from GitHub releases (MIT, v0.9.0)
# Pinned to specific release tag for reproducibility.
# Alternative: build from source at commit ee68144, but that requires build-essential
# and adds ~500MB to the image. Pre-built binary from releases is lighter.
RUN curl -L -o /usr/local/bin/cbm \
    https://github.com/DeusData/codebase-memory-mcp/releases/download/v0.9.0/codebase-memory-mcp-linux-x64 \
    && chmod +x /usr/local/bin/cbm

# Install Python dependencies for the bridge
RUN pip install --no-cache-dir fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3

# Copy bridge service code
WORKDIR /app
COPY main.py /app/

# Environment defaults (can be overridden in compose)
ENV CBM_BIN=/usr/local/bin/cbm
ENV CBM_CACHE_DIR=/data/cbm_cache
ENV CBM_TIMEOUT=300
ENV PORT=8200

# Create cache directory
RUN mkdir -p /data/cbm_cache

# Health check: hit the /health endpoint which does a real CBM tool call
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8200/health')"

# Run the bridge
CMD ["python", "main.py"]
