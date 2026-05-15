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


def _coerce_timestamp(value: object) -> Optional[str]:
    """Coerce a memory's ``created_at`` value to an ISO-8601 string.

    Accepts ``datetime`` or string. Returns ``None`` for anything else
    so the caller can fall back to ``datetime.now()``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value:
        return value
    return None


class ConversationCompilerExtension:
    """NeuralScape extension for automatic memory capture and compilation.

    Extracts facts from conversation turns, stores them in NeuralScape,
    writes human-readable logs to an Obsidian vault, and compiles
    daily logs into structured knowledge articles.
    """

    # Declare manifest at class level so issubclass_safe() detects it
    manifest: ExtensionManifest = ExtensionManifest(**_manifest_data)

    def __init__(self) -> None:
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
        elif event_type == "memory_stored":
            return await self._handle_memory_stored(payload)
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

    async def _handle_memory_stored(self, payload: dict) -> Optional[dict]:
        """Handle memory_stored: write shared memories to vault category folder + daily log.

        Private memories never reach the vault (multi-user privacy). For shared
        memories, the entry's date/time is taken from the memory's actual
        ``created_at`` so historical writes don't all collapse into today's log.
        Writes from the flush path are skipped to avoid double-writing.
        """
        if payload.get("source") == "conversation-compiler":
            return None

        content = payload.get("content", "")
        category = payload.get("category", "")
        project_id = payload.get("project_id")
        session_id = payload.get("run_id") or payload.get("agent_id") or "api"

        if not content or not category:
            return None

        from schemas import MemoryVisibility, default_visibility_for_category

        visibility_raw = payload.get("visibility")
        if visibility_raw is None:
            resolved_visibility = default_visibility_for_category(category)
        else:
            resolved_visibility = (
                visibility_raw
                if isinstance(visibility_raw, MemoryVisibility)
                else MemoryVisibility(visibility_raw)
            )
        if resolved_visibility != MemoryVisibility.SHARED:
            return None

        created_at_raw = payload.get("created_at")
        ts = _coerce_timestamp(created_at_raw) or datetime.now().isoformat()
        date = ts[:10]
        time_str = ts[11:16] if len(ts) >= 16 else datetime.now().strftime("%H:%M")

        try:
            cat_path = self.writer.append_category_entry(
                category=category,
                content=content,
                project_id=project_id,
                session_id=session_id,
                timestamp=ts,
            )
            self.writer.append_daily_log(date, [
                {
                    "time": time_str,
                    "category": category,
                    "content": content,
                    "session_id": session_id,
                }
            ])
            return {"vault_path": cat_path}
        except Exception:
            logger.exception("Failed to write memory to vault")
            return None

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
            result = await compile_date(today, self.service, self.writer, user_id=user_id)
            self.writer.append_log(f"Auto-compiled {today} on session end")
            return result.model_dump()
        except Exception:
            logger.exception("Auto-compile failed")
            return None

    async def _handle_compile_requested(self, payload: dict) -> Optional[dict]:
        """Handle explicit compile request."""
        date = payload.get("date")
        user_id = payload.get("user_id", "")
        if date:
            result = await compile_date(date, self.service, self.writer, user_id=user_id)
            return result.model_dump()
        else:
            results = await compile_all_pending(self.service, self.writer, user_id=user_id)
            return {"dates_compiled": len(results), "results": [r.model_dump() for r in results]}

    def get_routes(self) -> Optional[APIRouter]:
        """Return the API router for this extension."""
        return create_router(
            service=self.service,
            writer=self.writer,
            task_manager=self._task_manager,
        )
