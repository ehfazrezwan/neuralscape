"""CBM bridge service — owned REST shim over CBM's stdio MCP.

Spawns and keeps warm the CBM binary, exposing ONLY the structured JSON tools
we use (no raw query_graph/Cypher). The bridge owns the CBM lifecycle and
provides a clean HTTP surface for the NS stack.

Tools exposed:
  - index_repository: Index a repo (returns {project, nodes, edges, status}).
  - search_graph: Search symbols by name pattern.
  - trace_path: Trace call relationships (neighbors).
  - get_code_snippet: Fetch code snippets.
  - get_architecture: Get architecture overview.
  - delete_project: Delete an indexed project.
  - index_status: Health/staleness check.

Raw query_graph Cypher is BANNED (lexer quote-escape bug → silent WHERE drop).
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────

CBM_BIN = os.getenv("CBM_BIN", "/data/ice/tools/cbm/codebase-memory-mcp")
CBM_CACHE_DIR = os.getenv("CBM_CACHE_DIR", "/data/cbm_cache")
CBM_TIMEOUT = int(os.getenv("CBM_TIMEOUT", "300"))  # 5 minutes default


# ── Request/Response models ─────────────────────────────────────────


class IndexRepositoryRequest(BaseModel):
    repo_path: str = Field(description="Absolute path to the repository to index")


class IndexRepositoryResponse(BaseModel):
    project: str
    nodes: int
    edges: int
    status: str


class SearchGraphRequest(BaseModel):
    project: str = Field(description="Project slug from index_repository")
    name_pattern: str = Field(description="Symbol name pattern to search")


class SearchGraphResponse(BaseModel):
    results: list[dict] = Field(default_factory=list)


class TracePathRequest(BaseModel):
    project: str
    function_name: str
    direction: str = Field(default="both", description="incoming | outgoing | both")
    depth: int = Field(default=1, ge=1, le=10)


class TracePathResponse(BaseModel):
    paths: list[dict] = Field(default_factory=list)


class GetCodeSnippetRequest(BaseModel):
    project: str
    file_path: str
    start_line: int | None = None
    end_line: int | None = None


class GetCodeSnippetResponse(BaseModel):
    content: str
    file_path: str


class GetArchitectureRequest(BaseModel):
    project: str


class GetArchitectureResponse(BaseModel):
    architecture: str


class DeleteProjectRequest(BaseModel):
    project: str


class DeleteProjectResponse(BaseModel):
    status: str


class IndexStatusResponse(BaseModel):
    projects: list[dict] = Field(default_factory=list)
    cache_dir: str
    cbm_version: str | None = None


# ── CBM subprocess manager ──────────────────────────────────────────


class CBMManager:
    """Manages the CBM binary lifecycle and tool invocations.

    CBM is invoked via `cbm cli <tool> <json>` subprocess calls. Each call is
    independent (stateless CLI mode). The manager provides the environment and
    timeout wrapping.
    """

    def __init__(self, cbm_bin: str, cache_dir: str, timeout: int):
        self.cbm_bin = cbm_bin
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.version: str | None = None

        # Ensure cache dir exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Check CBM binary exists
        if not Path(cbm_bin).exists():
            raise FileNotFoundError(f"CBM binary not found: {cbm_bin}")

    def _env(self) -> dict:
        """Environment for CBM subprocess: pin cache dir, quiet logs."""
        env = dict(os.environ)
        env["CBM_CACHE_DIR"] = self.cache_dir
        env["CBM_LOG_LEVEL"] = "none"  # Logs go to stderr; stdout is clean JSON
        return env

    async def _get_version(self) -> str:
        """Get CBM version via --version."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cbm_bin,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(),
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=10
            )
            if proc.returncode == 0 and stdout:
                # Output: "codebase-memory-mcp 0.9.0"
                parts = stdout.decode().strip().split()
                return f"cbm@{parts[-1]}" if parts else "cbm@unknown"
        except Exception as e:
            logger.warning(f"Failed to get CBM version: {e}")
        return "cbm@unknown"

    async def call_tool(self, tool: str, args: dict) -> dict:
        """Call a CBM CLI tool: `cbm cli <tool> <json>`.

        Args:
            tool: Tool name (index_repository, search_graph, etc.).
            args: Tool arguments as dict (serialized to JSON).

        Returns:
            Parsed JSON response from CBM stdout.

        Raises:
            HTTPException: On subprocess failure or timeout.
        """
        cmd = [self.cbm_bin, "cli", tool, json.dumps(args)]

        logger.info(f"CBM tool call: {tool} with args: {args}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(),
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"CBM tool {tool} failed: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=f"CBM tool {tool} failed: {error_msg}"
                )

            # Parse JSON from stdout
            try:
                result = json.loads(stdout.decode().strip())
                logger.info(f"CBM tool {tool} succeeded")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"CBM tool {tool} returned invalid JSON: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"CBM returned invalid JSON: {e}"
                )

        except asyncio.TimeoutError:
            logger.error(f"CBM tool {tool} timed out after {self.timeout}s")
            raise HTTPException(
                status_code=504,
                detail=f"CBM tool {tool} timed out after {self.timeout}s"
            )
        except Exception as e:
            logger.exception(f"CBM tool {tool} failed unexpectedly")
            raise HTTPException(
                status_code=500,
                detail=f"CBM tool {tool} failed: {str(e)}"
            )


# ── FastAPI app ─────────────────────────────────────────────────────


cbm_manager: CBMManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize CBM manager on startup."""
    global cbm_manager
    cbm_manager = CBMManager(CBM_BIN, CBM_CACHE_DIR, CBM_TIMEOUT)
    cbm_manager.version = await cbm_manager._get_version()
    logger.info(f"CBM bridge started with {cbm_manager.version}")
    yield
    logger.info("CBM bridge shutting down")


app = FastAPI(
    title="CBM Bridge Service",
    description="REST shim over CBM's stdio MCP (structured tools only, no raw Cypher)",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health endpoint ─────────────────────────────────────────────────


@app.get("/health")
async def health() -> JSONResponse:
    """Health check: performs a real cheap tool call (list_projects).

    Returns 200 OK if CBM is reachable and can execute a tool call.
    This is NOT just a TCP ping — it verifies CBM actually works.
    """
    if cbm_manager is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "CBM manager not initialized"},
        )

    try:
        # Real tool call: list_projects is cheap and always succeeds
        result = await cbm_manager.call_tool("list_projects", {})
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "cbm_version": cbm_manager.version,
                "projects_count": len(result.get("projects", [])),
            },
        )
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "unreachable", "detail": str(e)},
        )


# ── Tool endpoints ──────────────────────────────────────────────────


@app.post("/index_repository", response_model=IndexRepositoryResponse)
async def index_repository(req: IndexRepositoryRequest) -> IndexRepositoryResponse:
    """Index a repository with CBM."""
    result = await cbm_manager.call_tool("index_repository", {"repo_path": req.repo_path})
    return IndexRepositoryResponse(**result)


@app.post("/search_graph", response_model=SearchGraphResponse)
async def search_graph(req: SearchGraphRequest) -> SearchGraphResponse:
    """Search the code graph by name pattern."""
    result = await cbm_manager.call_tool(
        "search_graph",
        {"project": req.project, "name_pattern": req.name_pattern},
    )
    return SearchGraphResponse(results=result.get("results", []))


@app.post("/trace_path", response_model=TracePathResponse)
async def trace_path(req: TracePathRequest) -> TracePathResponse:
    """Trace call paths (neighbors)."""
    result = await cbm_manager.call_tool(
        "trace_path",
        {
            "project": req.project,
            "function_name": req.function_name,
            "direction": req.direction,
            "depth": req.depth,
        },
    )
    return TracePathResponse(paths=result.get("paths", []))


@app.post("/get_code_snippet", response_model=GetCodeSnippetResponse)
async def get_code_snippet(req: GetCodeSnippetRequest) -> GetCodeSnippetResponse:
    """Get code snippet from a file."""
    args = {"project": req.project, "file_path": req.file_path}
    if req.start_line is not None:
        args["start_line"] = req.start_line
    if req.end_line is not None:
        args["end_line"] = req.end_line

    result = await cbm_manager.call_tool("get_code_snippet", args)
    return GetCodeSnippetResponse(**result)


@app.post("/get_architecture", response_model=GetArchitectureResponse)
async def get_architecture(req: GetArchitectureRequest) -> GetArchitectureResponse:
    """Get architecture overview."""
    result = await cbm_manager.call_tool("get_architecture", {"project": req.project})
    return GetArchitectureResponse(architecture=result.get("architecture", ""))


@app.post("/delete_project", response_model=DeleteProjectResponse)
async def delete_project(req: DeleteProjectRequest) -> DeleteProjectResponse:
    """Delete an indexed project."""
    result = await cbm_manager.call_tool("delete_project", {"project": req.project})
    return DeleteProjectResponse(status=result.get("status", "deleted"))


@app.get("/index_status", response_model=IndexStatusResponse)
async def index_status() -> IndexStatusResponse:
    """Get indexing status (list of projects, health/staleness check)."""
    result = await cbm_manager.call_tool("list_projects", {})
    return IndexStatusResponse(
        projects=result.get("projects", []),
        cache_dir=cbm_manager.cache_dir,
        cbm_version=cbm_manager.version,
    )


# ── Main entry point ────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8200"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
