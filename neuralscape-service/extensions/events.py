"""Standard event types and schemas for the NeuralScape extension system.

Defines the canonical event types that NeuralScape core emits and the
Pydantic models for their payloads. Extensions declare which event types
they listen to via their manifest.hooks list.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Standard event types emitted by NeuralScape core."""

    CONVERSATION_TURN = "conversation_turn"
    """A conversation turn to process (messages from a user/agent session)."""

    SESSION_START = "session_start"
    """A new agent session began."""

    SESSION_END = "session_end"
    """An agent session ended."""

    MEMORY_STORED = "memory_stored"
    """A memory was successfully stored (useful for post-processing extensions)."""

    COMPILE_REQUESTED = "compile_requested"
    """Daily compilation/summarization requested."""


class ConversationTurnEvent(BaseModel):
    """Payload for conversation_turn events."""

    user_id: str
    messages: list[dict] = Field(description="Conversation messages ({role, content} dicts)")
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None


class SessionStartEvent(BaseModel):
    """Payload for session_start events."""

    user_id: str
    session_id: str
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Optional[dict] = None


class SessionEndEvent(BaseModel):
    """Payload for session_end events."""

    user_id: str
    session_id: str
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    duration_seconds: Optional[float] = None


class MemoryStoredEvent(BaseModel):
    """Payload for memory_stored events."""

    user_id: str
    memory_id: str
    content: str
    category: Optional[str] = None
    scope: Optional[str] = None
    project_id: Optional[str] = None


class CompileRequestedEvent(BaseModel):
    """Payload for compile_requested events."""

    user_id: str
    project_id: Optional[str] = None
    requested_by: Optional[str] = Field(
        default=None,
        description="Who/what triggered the compilation (e.g. 'cron', 'manual', 'api')",
    )


# Map event types to their payload models for validation
EVENT_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    EventType.CONVERSATION_TURN: ConversationTurnEvent,
    EventType.SESSION_START: SessionStartEvent,
    EventType.SESSION_END: SessionEndEvent,
    EventType.MEMORY_STORED: MemoryStoredEvent,
    EventType.COMPILE_REQUESTED: CompileRequestedEvent,
}
