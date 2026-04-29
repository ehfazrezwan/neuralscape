"""API routes for the conversation-compiler extension.

Mounted at /v1/extensions/conversation-compiler/ by the extension registry.
"""

from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException

from memory_service import MemoryService
from task_manager import TaskManager

from .compile import compile_all_pending, compile_date
from .config import compiler_settings
from .flush import flush_conversation_turn
from .lint import run_lint
from .obsidian_writer import ObsidianWriter
from .query import query_knowledge_base
from .schemas import (
    CompileRequest,
    CompileResult,
    FlushRequest,
    FlushResult,
    LintRequest,
    LintResult,
    QueryRequest,
    QueryResult,
    StatusResponse,
)

logger = structlog.get_logger(__name__)


def create_router(
    service: MemoryService,
    writer: ObsidianWriter,
    task_manager: TaskManager | None = None,
) -> APIRouter:
    """Create the API router for the conversation-compiler extension.

    Args:
        service: MemoryService instance for memory storage.
        writer: ObsidianWriter instance for vault I/O.
        task_manager: Optional TaskManager for async job enqueuing.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter()

    @router.post("/flush", status_code=202, response_model=dict)
    async def flush(request: FlushRequest) -> dict:
        """Submit a conversation turn for fact extraction.

        Enqueues the extraction to the ARQ worker for async processing.
        Returns 202 Accepted with a task reference.
        """
        timestamp = request.timestamp or datetime.now().isoformat()

        if task_manager and task_manager.pool:
            try:
                job = await task_manager.pool.enqueue_job(
                    "process_conversation_flush",
                    request.user_message,
                    request.assistant_response,
                    request.session_id,
                    request.channel,
                    timestamp,
                    request.project_id,
                    request.user_id,
                )
                task_id = job.job_id if job else "duplicate"
                return {"status": "accepted", "task_id": task_id}
            except Exception:
                logger.warning("ARQ enqueue failed, falling back to sync flush")

        # Fallback: run synchronously
        result = await flush_conversation_turn(
            user_message=request.user_message,
            assistant_response=request.assistant_response,
            session_id=request.session_id,
            channel=request.channel,
            timestamp=timestamp,
            project_id=request.project_id,
            user_id=request.user_id,
            service=service,
            writer=writer,
        )
        return {
            "status": "completed",
            "result": result.model_dump(),
        }

    @router.post("/compile", status_code=202, response_model=dict)
    async def compile(request: CompileRequest) -> dict:
        """Trigger compilation for a specific date or all pending.

        Enqueues compilation to the ARQ worker for async processing.
        """
        if task_manager and task_manager.pool:
            try:
                job = await task_manager.pool.enqueue_job(
                    "process_conversation_compile",
                    request.date,
                    request.user_id,
                )
                task_id = job.job_id if job else "duplicate"
                return {"status": "accepted", "task_id": task_id}
            except Exception:
                logger.warning("ARQ enqueue failed, falling back to sync compile")

        # Fallback: run synchronously
        if request.date:
            result = await compile_date(request.date, service, writer, user_id=request.user_id)
            return {"status": "completed", "result": result.model_dump()}
        else:
            results = await compile_all_pending(service, writer, user_id=request.user_id)
            return {
                "status": "completed",
                "results": [r.model_dump() for r in results],
            }

    @router.post("/query", response_model=QueryResult)
    async def query(request: QueryRequest) -> QueryResult:
        """Query the knowledge base using index-guided retrieval.

        Synchronous — returns the answer directly.
        """
        return await query_knowledge_base(
            question=request.question,
            writer=writer,
            file_back=request.file_back,
        )

    @router.post("/lint", response_model=LintResult)
    async def lint(request: LintRequest) -> LintResult:
        """Run health checks on the Obsidian vault."""
        return await run_lint(
            writer=writer,
            structural_only=request.structural_only,
        )

    @router.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        """Get extension status and stats."""
        all_files = writer.list_all_files()
        daily_logs = writer.list_daily_logs()

        # Find last flush and compile times from the log
        log_content = writer.read_file("log.md")
        last_flush = None
        last_compile = None
        if log_content:
            # Parse log entries for timestamps
            flush_matches = list(
                __import__("re").finditer(
                    r"\*\*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\*\* — (?:Flush|flush)",
                    log_content,
                )
            )
            compile_matches = list(
                __import__("re").finditer(
                    r"\*\*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\*\* — Compiled",
                    log_content,
                )
            )
            if flush_matches:
                last_flush = flush_matches[-1].group(1)
            if compile_matches:
                last_compile = compile_matches[-1].group(1)

        # Count articles (exclude daily logs, index, log)
        article_count = len(
            [f for f in all_files if not f.startswith("Daily/") and f not in ("index.md", "log.md")]
        )

        return StatusResponse(
            last_flush=last_flush,
            last_compile=last_compile,
            article_count=article_count,
            daily_log_count=len(daily_logs),
            vault_path=str(compiler_settings.vault_path),
        )

    return router
