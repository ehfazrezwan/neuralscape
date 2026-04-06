"""Conversation Compiler — NeuralScape Extension.

Implements Karpathy's "LLM Wiki" pattern + coleam00's "Claude Memory Compiler"
concept as a NeuralScape extension. Automatically captures facts from conversations,
writes to daily logs, and compiles into structured knowledge articles.

Hooks:
    - conversation_turn: Extract facts from a conversation turn
    - session_end: Flush remaining context + check if compile needed
    - compile_requested: Trigger daily compilation
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter

from extensions.base import ExtensionManifest, NeuralscapeExtension
from memory_service import MemoryService

from .compile import compile_all_pending, compile_date
from .config import compiler_settings
from .flush import flush_conversation_turn
from .obsidian_writer import ObsidianWriter
from .routes import create_router

logger = structlog.get_logger(__name__)

# Load manifest from JSON file
_MANIFEST_PATH = Path(__file__).parent / "manifest.json"
_manifest_data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


class ConversationCompilerExtension:
    """NeuralScape extension for automatic memory capture and compilation.

    Extracts facts from conversation turns, stores them in NeuralScape,
    writes human-readable logs to an Obsidian vault, and compiles
    daily logs into structured knowledge articles.
    """

    def __init__(self) -> None:
        self.manifest = ExtensionManifest(**_manifest_data)
        self._service: Optional[MemoryService] = None
        self._writer: Optional[ObsidianWriter] = None
        self._task_manager = None

    @property
    def service(self) -> MemoryService:
        if self._service is None:
            self._service = MemoryService()
        return self._service

    @property
    def writer(self) -> ObsidianWriter:
        if self._writer is None:
            self._writer = ObsidianWriter()
        return self._writer

    async def startup(self) -> None:
        """Initialize the extension: create writer, warm up service."""
        logger.info(
            "Conversation Compiler starting up",
            vault=str(compiler_settings.vault_path),
            auto_compile=compiler_settings.auto_compile,
            compile_after_hour=compiler_settings.compile_after_hour,
        )

        # Initialize writer (creates vault dirs if needed)
        self._writer = ObsidianWriter()

        # Initialize memory service
        self._service = MemoryService()

        # Try to get task manager from the main app (set by mount_routes)
        # Will be set when routes are created

        logger.info("Conversation Compiler ready")

    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Conversation Compiler shutting down")
        # MemoryService cleanup is handled by the main app

    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
        """Handle events from NeuralScape core.

        Args:
            event_type: One of 'conversation_turn', 'session_end', 'compile_requested'.
            payload: Event-specific data.

        Returns:
            Optional dict with results.
        """
        if event_type == "conversation_turn":
            return await self._handle_conversation_turn(payload)
        elif event_type == "session_end":
            return await self._handle_session_end(payload)
        elif event_type == "compile_requested":
            return await self._handle_compile_requested(payload)
        return None

    async def _handle_conversation_turn(self, payload: dict) -> Optional[dict]:
        """Extract facts from a conversation turn."""
        messages = payload.get("messages", [])
        user_id = payload.get("user_id", "")
        project_id = payload.get("project_id")

        if not messages or not user_id:
            return None

        # Extract user message and assistant response from messages
        user_message = ""
        assistant_response = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_message = content
            elif role == "assistant":
                assistant_response = content

        if not user_message:
            return None

        session_id = payload.get("run_id") or payload.get("agent_id") or "event"

        result = await flush_conversation_turn(
            user_message=user_message,
            assistant_response=assistant_response,
            session_id=session_id,
            channel="event",
            timestamp=None,
            project_id=project_id,
            user_id=user_id,
            service=self.service,
            writer=self.writer,
        )

        return result.model_dump() if result.facts_extracted > 0 else None

    async def _handle_session_end(self, payload: dict) -> Optional[dict]:
        """Handle session end: check if auto-compile is needed."""
        user_id = payload.get("user_id", "")
        if not user_id:
            return None

        # Check if auto-compile should run
        if not compiler_settings.auto_compile:
            return None

        now = datetime.now()
        if now.hour < compiler_settings.compile_after_hour:
            return None

        # Check if today's log has uncompiled entries
        today = now.strftime("%Y-%m-%d")
        if self.writer.is_daily_log_compiled(today):
            return None

        logger.info("Auto-compile triggered on session end", date=today)
        try:
            result = await compile_date(today, self.service, self.writer)
            self.writer.append_log(f"Auto-compiled {today} on session end")
            return result.model_dump()
        except Exception:
            logger.exception("Auto-compile failed")
            return None

    async def _handle_compile_requested(self, payload: dict) -> Optional[dict]:
        """Handle explicit compile request."""
        date = payload.get("date")
        if date:
            result = await compile_date(date, self.service, self.writer)
            return result.model_dump()
        else:
            results = await compile_all_pending(self.service, self.writer)
            return {"dates_compiled": len(results), "results": [r.model_dump() for r in results]}

    def get_routes(self) -> Optional[APIRouter]:
        """Return the API router for this extension."""
        return create_router(
            service=self.service,
            writer=self.writer,
            task_manager=self._task_manager,
        )
