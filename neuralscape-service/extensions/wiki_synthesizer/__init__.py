"""Wiki Synthesizer — NeuralScape Extension.

Periodically reads shared memories and produces topical wiki pages in
the Obsidian vault. See ``synthesizer.py`` for the orchestration logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter

from extensions.base import ExtensionManifest
from memory_service import MemoryService

from .config import synthesizer_settings
from .synthesizer import synthesize_all

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent / "manifest.json"
_manifest_data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


class WikiSynthesizerExtension:
    """Cron-driven wiki page generator.

    Implements the NeuralScape extension protocol: a manifest, lifecycle
    hooks, and (optionally) routes. The actual synthesis lives in
    :mod:`extensions.wiki_synthesizer.synthesizer` so it can be invoked
    directly from the ARQ cron without going through the event bus.
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
            "Wiki Synthesizer starting up",
            extra={
                "enabled": synthesizer_settings.enabled,
                "cron_hours": synthesizer_settings.cron_hours,
                "wiki_dir": str(synthesizer_settings.wiki_dir),
            },
        )
        self._service = MemoryService()

    async def shutdown(self) -> None:
        logger.info("Wiki Synthesizer shutting down")

    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
        # This extension intentionally subscribes to no events — synthesis
        # is driven by the ARQ cron (and the admin endpoint), not events.
        return None

    async def run_synthesis(
        self,
        *,
        only_category: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Trigger one synthesis pass. Called by the cron and the admin endpoint."""
        result = await synthesize_all(
            service=self.service,
            settings=synthesizer_settings,
            only_category=only_category,
            dry_run=dry_run,
        )
        return {
            "pages_created": result.pages_created,
            "pages_updated": result.pages_updated,
            "memories_processed": result.memories_processed,
            "communities_skipped_empty": result.communities_skipped_empty,
            "errors": result.errors,
            "pages": [
                {
                    "category": p.category,
                    "community_id": p.community_id,
                    "wiki_path": p.wiki_path,
                    "created": p.created,
                    "source_memory_count": p.source_memory_count,
                }
                for p in result.pages
            ],
        }

    def get_routes(self) -> Optional[APIRouter]:
        # v1 surfaces the synthesizer via a core admin endpoint in main.py
        # rather than an extension-scoped /v1/extensions/wiki-synthesizer/
        # route, so we return None here.
        return None
