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

from fastapi import APIRouter
from pydantic import BaseModel, Field

from extensions.base import ExtensionManifest
from memory_service import MemoryService

from .config import dreaming_settings
from .sweep import dream_all, get_last_run

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

        @router.post("/run")
        async def run_dream(req: DreamRequest):
            """Trigger one dreaming sweep (synchronous; use dry_run to tune)."""
            if not dreaming_settings.enabled and not req.force:
                return {
                    "run_id": None,
                    "pools": [],
                    "error": "DREAMING_ENABLED=false — set the env var (or force=true) to run",
                }
            run = await dream_all(
                service=self.service,
                settings=dreaming_settings,
                dry_run=req.dry_run,
                only_pool=req.pool,
                force=req.force,
            )
            return run.to_dict()

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
                "dreams_dir": str(dreaming_settings.dreams_dir),
                "last_run": get_last_run(),
            }

        return router
