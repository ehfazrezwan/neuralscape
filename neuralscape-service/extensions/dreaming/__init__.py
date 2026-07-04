"""Dreaming — Neuralscape Extension.

Background memory consolidation (the successor to the wiki_synthesizer):
a cron-driven light → deep → REM sweep per memory pool that merges
duplicates, invalidates contradictions bi-temporally, prunes noise,
reframes stale tenses, and writes new reflection insights back into the
store as first-class recallable memories. See docs/DREAMING_MODE_SPEC.md
and ``sweep.py`` for the orchestration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from extensions.base import ExtensionManifest
from memory_service import MemoryService

from .config import dreaming_settings
from .sweep import get_last_run

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent / "manifest.json"
_manifest_data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


class DreamRequest(BaseModel):
    dry_run: bool = Field(
        default=False,
        description="Plan and report every action without writing anything.",
    )
    pool: str | None = Field(
        default=None,
        description="Restrict the sweep to one pool key (e.g. 'shared', 'user--alice').",
    )
    force: bool = Field(
        default=False,
        description="Bypass the time/volume gates (never the pool lock).",
    )


class DreamingExtension:
    """Cron-driven memory consolidation ("dreaming").

    Implements the Neuralscape extension protocol. The sweep itself lives
    in :mod:`extensions.dreaming.sweep` so the ARQ cron can invoke it
    without the event bus; the routes here are the admin surface.
    """

    manifest: ExtensionManifest = ExtensionManifest(**_manifest_data)

    def __init__(self) -> None:
        self._service: Optional[MemoryService] = None

    @property
    def service(self) -> MemoryService:
        if self._service is None:
            self._service = MemoryService()
        return self._service

    async def startup(self) -> None:
        logger.info(
            "Dreaming extension starting up",
            extra={
                "enabled": dreaming_settings.enabled,
                "cron_hours": dreaming_settings.cron_hours,
                "dreams_dir": str(dreaming_settings.dreams_dir),
            },
        )
        self._service = MemoryService()

    async def shutdown(self) -> None:
        logger.info("Dreaming extension shutting down")

    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
        # Sweeps are cron/admin-driven, not event-driven.
        return None

    def get_routes(self) -> Optional[APIRouter]:
        router = APIRouter(tags=["dreaming"])

        @router.post("/run", status_code=202)
        async def run_dream(req: DreamRequest):
            """Enqueue one dreaming sweep onto the graph worker (202 + poll).

            Never sweeps in-process: a sweep is minutes of LLM + store
            work, and on the API's event loop it starves /health until
            autoheal restarts the container (observed live 2026-07-03).
            Poll GET /status for the DreamRun result.
            """
            if not dreaming_settings.enabled and not req.force:
                raise HTTPException(
                    status_code=409,
                    detail="DREAMING_ENABLED=false — set the env var (or force=true) to run",
                )
            from arq import create_pool

            from config import parse_redis_settings, settings as core_settings

            arq_pool = await create_pool(parse_redis_settings())
            try:
                job = await arq_pool.enqueue_job(
                    "run_dream_sweep",
                    req.pool,
                    req.dry_run,
                    req.force,
                    _queue_name=core_settings.graph_queue_name,
                )
            finally:
                await arq_pool.close()
            return {
                "job_id": job.job_id if job else None,
                "status": "enqueued",
                "poll": "/v1/extensions/dreaming/status",
            }

        @router.get("/status")
        async def dream_status():
            """Config + the last DreamRun (cross-process, from Redis)."""
            return {
                "enabled": dreaming_settings.enabled,
                "cron_hours": dreaming_settings.cron_hours,
                "min_hours": dreaming_settings.min_hours,
                "min_new_memories": dreaming_settings.min_new_memories,
                "auto_apply_confidence": dreaming_settings.auto_apply_confidence,
                "prune_strength_threshold": dreaming_settings.prune_strength_threshold,
                "reflection_enabled": dreaming_settings.reflection_enabled,
                "bridges_enabled": dreaming_settings.bridges_enabled,
                "identity_card_enabled": dreaming_settings.identity_card_enabled,
                "dreams_dir": str(dreaming_settings.dreams_dir),
                "last_run": get_last_run(),
            }

        @router.get("/card")
        async def get_identity_card(pool: str):
            """The pinned identity card for one pool (B4).

            ``pool`` is a pool key (``user--<uid>`` or
            ``shared--project--<pid>``). Returns the grammar-constrained
            lines for session-start injection; 404 when the sweep hasn't
            produced a card for that pool yet. Cards are pinned artifacts
            in Redis — never searchable memories.
            """
            import asyncio

            from auth import current_user_id
            from config import settings as core_settings

            from .card import card_read_allowed, load_card
            from .sweep import _get_redis

            caller = current_user_id.get()
            if not card_read_allowed(
                pool, caller, is_dictator=core_settings.is_dictator(caller)
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Another user's private card is not readable",
                )
            data = await asyncio.to_thread(load_card, _get_redis(), pool)
            lines = (data or {}).get("lines") or []
            if not lines:
                raise HTTPException(
                    status_code=404, detail=f"No identity card for pool {pool!r}"
                )
            return {
                "status": "ok",
                "pool": pool,
                "lines": lines,
                "card": "\n".join(lines),
                "updated_at": (data or {}).get("updated_at"),
            }

        return router
