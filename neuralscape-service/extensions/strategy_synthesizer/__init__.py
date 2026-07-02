"""Strategy Synthesizer — NeuralScape Extension.

Periodically merges a trading strategy's ingested rule memories into one
canonical, versioned playbook page (Obsidian vault ``Playbooks/`` tree). Grouped
by strategy; non-destructive. See ``synthesizer.py`` for orchestration.

Cron-driven (empty hooks) — the pass is invoked from the ARQ cron in worker.py,
not the event bus. This module exposes the extension class for the
ExtensionRegistry's lifecycle management.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter

from extensions.base import ExtensionManifest
from memory_service import MemoryService

from .config import strategy_synthesizer_settings
from .synthesizer import synthesize_all

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent / "manifest.json"
_manifest_data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


class StrategySynthesizerExtension:
    """Cron-driven trading-strategy playbook generator."""

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
            "Strategy Synthesizer starting up",
            extra={
                "enabled": strategy_synthesizer_settings.enabled,
                "cron_hours": strategy_synthesizer_settings.cron_hours,
                "playbook_dir": str(strategy_synthesizer_settings.playbook_dir),
            },
        )
        self._service = MemoryService()

    async def shutdown(self) -> None:
        logger.info("Strategy Synthesizer shutting down")

    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
        # No event subscriptions — synthesis is cron-driven.
        return None

    async def run_synthesis(
        self, *, only_strategy: str | None = None, dry_run: bool = False, force: bool = False
    ) -> dict:
        """Trigger one synthesis pass (called by the cron / admin tooling)."""
        result = await synthesize_all(
            service=self.service,
            settings=strategy_synthesizer_settings,
            only_strategy=only_strategy,
            dry_run=dry_run,
            force=force,
        )
        return {
            "playbooks_created": result.playbooks_created,
            "playbooks_updated": result.playbooks_updated,
            "playbooks_skipped_unchanged": result.playbooks_skipped_unchanged,
            "memories_processed": result.memories_processed,
            "errors": result.errors,
            "playbooks": [
                {
                    "strategy_name": p.strategy_name,
                    "owner": p.owner,
                    "playbook_path": p.playbook_path,
                    "created": p.created,
                    "source_memory_count": p.source_memory_count,
                }
                for p in result.playbooks
            ],
        }

    def get_routes(self) -> Optional[APIRouter]:
        return None
