# Precise Neighbors Resolver Service Dockerfile
#
# A thin FastAPI REST shim over `pyright-langserver` (pyright, MIT). The shim
# owns the language-server subprocess lifecycle (one warm server per repo) and
# resolves a repo's call sites to their real definitions at INDEX time.
#
# This image is INTENTIONALLY separate from the neuralscape-service image: the
# LSP/Node toolchain lives here, so the main service image (and its container
# gate) is never touched. That isolation is the whole point of the service.
#
# Third-party (both MIT): pyright (Microsoft) — see THIRD_PARTY.md.

# Node base: pyright-langserver is a Node program; python3 is layered on top for
# the FastAPI shim. (Doing it this way is lighter than a python base + a full
# Node install, and keeps the pyright runtime canonical.)
FROM node:20-slim

# Python + build essentials for the shim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned pyright (provides `pyright` + `pyright-langserver`). MIT.
# The install must succeed (no `|| true` swallowing failures — that could ship an
# image without pyright that only fails at runtime); a separate check asserts the
# langserver binary is actually on PATH so a broken image fails the BUILD.
ARG PYRIGHT_VERSION=1.1.411
RUN npm install -g pyright@${PYRIGHT_VERSION}
RUN command -v pyright-langserver >/dev/null 2>&1 || (echo "pyright-langserver missing after install" && exit 1)

# Isolated venv for the shim (avoids PEP-668 externally-managed conflicts).
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"
RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    "uvicorn[standard]"==0.32.1 \
    pydantic==2.10.3 \
    anyio==4.6.2

WORKDIR /app
COPY lsp_client.py main.py /app/

# Environment defaults (overridable in compose).
ENV PORT=8201
ENV PYRIGHT_LANGSERVER=pyright-langserver
ENV PYRIGHT_SETTLE_SECONDS=2.0
ENV RESOLVER_TIMEOUT=600

# Health: verify the langserver binary is present + the shim is serving.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8201/health')"

CMD ["python3", "main.py"]
