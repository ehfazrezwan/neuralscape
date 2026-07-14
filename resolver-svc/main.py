"""Precise-neighbors resolver service — REST shim over pyright-langserver.

Mirrors the CBM-bridge pattern: the shim owns the language-server subprocess
lifecycle (one warm ``pyright-langserver`` per repo) and exposes exactly what NS
needs at INDEX time — resolve a repo's call sites to their real definitions.

Endpoint contract (index-time only; never on the interactive query path):

  POST /resolve_calls
    {"repo_path": "/abs/repo",
     "files": [{"path": "pkg/mod.py", "sites": [[line1based, col], ...]}, ...]}
  → {"repo_path": "...",
     "files": [{"path": "pkg/mod.py",
                "defs": [[def_abs_path | null, def_line_1based | null], ...]}, ...],
     "resolved": N, "total": M}

The per-site ``defs`` list is parallel to the request's ``sites`` and carries the
SAME shape as NS's in-process ``JediCallResolver.resolve_file`` return value, so
the NS driver (``LspCallResolver``) is a drop-in over REST and every downstream
step — span→FQN mapping, no-phantom MATCH-only store, dedup, stale-edge cleanup —
is reused unchanged. Definitions outside ``repo_path`` (stdlib/external) come back
as ``[null, null]`` and the caller drops them (no phantom minting).

Why pyright, not multilspy's Python backend: see ``lsp_client.py`` — multilspy
wraps jedi-language-server (= the Jedi fidelity NS already ships); pyright is the
type-checker-grade resolver this mission actually needs.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from lsp_client import LSPError, PyrightServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# A resolve request for a whole repo may open many files; give pyright room.
RESOLVE_TIMEOUT = int(os.getenv("RESOLVER_TIMEOUT", "600"))


# ── Request/response models ─────────────────────────────────────────


class FileSites(BaseModel):
    path: str = Field(description="Repo-relative file path")
    sites: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Call sites as [line (1-based), col (0-based)] pairs",
    )


class ResolveCallsRequest(BaseModel):
    repo_path: str = Field(description="Absolute path to the mounted repo root")
    files: list[FileSites] = Field(default_factory=list)


class FileDefs(BaseModel):
    path: str
    defs: list[tuple[str | None, int | None]] = Field(default_factory=list)


class ResolveCallsResponse(BaseModel):
    repo_path: str
    files: list[FileDefs] = Field(default_factory=list)
    resolved: int = 0
    total: int = 0


# ── Warm-server pool (one pyright per repo root) ─────────────────────


class ServerPool:
    """Caches one warm :class:`PyrightServer` per repo root.

    Warming pyright (spawn + initialize + first analysis pass) is the dominant
    cost; reusing it across every file in a repo's index amortizes it. A dead
    server (crash) is transparently replaced on next use.
    """

    def __init__(self):
        self._servers: dict[str, PyrightServer] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()  # guards the dicts only (fast)

    def _lock_for(self, root: str) -> threading.Lock:
        with self._registry_lock:
            lk = self._locks.get(root)
            if lk is None:
                lk = threading.Lock()
                self._locks[root] = lk
            return lk

    def get(self, repo_path: str) -> PyrightServer:
        root = str(Path(repo_path).resolve())
        # Per-root lock so a slow pyright warmup/restart for one repo does NOT
        # serialize resolve requests for OTHER repos (the registry lock is only
        # held briefly to fetch/create the per-root lock).
        with self._lock_for(root):
            srv = self._servers.get(root)
            if srv is not None and srv.is_alive():
                return srv
            if srv is not None:
                logger.warning("pyright for %s died — restarting", root)
                srv.stop()
            srv = PyrightServer(root)
            srv.start()
            srv.settle()
            with self._registry_lock:
                self._servers[root] = srv
            return srv

    def shutdown(self) -> None:
        with self._registry_lock:
            servers = list(self._servers.values())
            self._servers.clear()
        for srv in servers:
            srv.stop()


pool: ServerPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = ServerPool()
    logger.info("resolver-svc started (pyright LSP shim)")
    yield
    if pool is not None:
        pool.shutdown()
    logger.info("resolver-svc shutting down")


app = FastAPI(
    title="Precise Neighbors Resolver",
    description="REST shim over pyright-langserver: index-time call-graph resolution",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ───────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness: verifies the pyright-langserver binary is invokable.

    Does NOT warm a repo (no workspace yet at boot) — a version probe is the
    honest 'the toolchain is present and runnable' signal.
    """
    import shutil

    bin_name = os.getenv("PYRIGHT_LANGSERVER", "pyright-langserver")
    if shutil.which(bin_name) is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": f"{bin_name} not on PATH"},
        )
    return JSONResponse(status_code=200, content={"status": "ok", "langserver": bin_name})


# ── Resolve ──────────────────────────────────────────────────────────


def _resolve_repo(req: ResolveCallsRequest) -> ResolveCallsResponse:
    """Blocking resolve of every site in the request (runs in a worker thread)."""
    assert pool is not None
    repo_root = str(Path(req.repo_path).resolve())
    if not Path(repo_root).is_dir():
        raise HTTPException(status_code=400, detail=f"repo_path not a dir: {repo_root}")

    srv = pool.get(repo_root)
    out_files: list[FileDefs] = []
    resolved = 0
    total = 0

    repo_root_resolved = Path(repo_root).resolve()
    for f in req.files:
        abs_path = str((Path(repo_root) / f.path).resolve())
        # Path-traversal guard: a request must not use `../`/symlinks to make the
        # service read files outside the mounted repo (pyright didOpen reads file
        # contents). Anything escaping repo_root → all sites unresolved.
        try:
            Path(abs_path).relative_to(repo_root_resolved)
        except ValueError:
            logger.warning("rejecting out-of-repo file path: %s", f.path)
            out_files.append(FileDefs(path=f.path, defs=[(None, None)] * len(f.sites)))
            total += len(f.sites)
            continue
        defs: list[tuple[str | None, int | None]] = []
        for line1, col in f.sites:
            total += 1
            try:
                # tree-sitter/Jedi convention (1-based line, 0-based col) → LSP
                # 0-based position. Col is a byte offset upstream; on ASCII text
                # that equals the UTF-16 char offset LSP wants (documented caveat).
                hits = srv.request_definition(abs_path, line1 - 1, col)
            except LSPError:
                logger.debug("definition failed at %s:%d:%d", f.path, line1, col, exc_info=True)
                defs.append((None, None))
                continue
            picked: tuple[str | None, int | None] = (None, None)
            for def_path, def_line in hits:
                # Only in-repo definitions become edges; external → dropped.
                try:
                    Path(def_path).resolve().relative_to(Path(repo_root))
                except (ValueError, OSError):
                    continue
                picked = (def_path, def_line)
                break
            if picked[0] is not None:
                resolved += 1
            defs.append(picked)
        out_files.append(FileDefs(path=f.path, defs=defs))

    return ResolveCallsResponse(
        repo_path=repo_root, files=out_files, resolved=resolved, total=total
    )


@app.post("/resolve_calls", response_model=ResolveCallsResponse)
async def resolve_calls(req: ResolveCallsRequest) -> ResolveCallsResponse:
    """Resolve every call site in a repo to its in-repo definition (pyright)."""
    import anyio

    try:
        # pyright I/O is blocking stdio — run off the event loop.
        return await anyio.to_thread.run_sync(_resolve_repo, req)
    except HTTPException:
        raise
    except LSPError as e:
        logger.exception("resolve_calls LSP fault")
        raise HTTPException(status_code=502, detail=f"pyright fault: {e}")
    except Exception as e:  # noqa: BLE001
        logger.exception("resolve_calls unexpected fault")
        raise HTTPException(status_code=500, detail=f"resolver fault: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8201"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
