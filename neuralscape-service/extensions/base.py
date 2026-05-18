"""Extension protocol and manifest for NeuralScape extensions.

Defines the interface that all NeuralScape extensions must implement.
Extensions are self-contained packages that can add API routes, hook into
events, and extend NeuralScape's capabilities without touching core code.
"""

from typing import Optional, Protocol, runtime_checkable

from fastapi import APIRouter
from pydantic import BaseModel, Field


class ExtensionManifest(BaseModel):
    """Metadata describing a NeuralScape extension."""

    name: str = Field(
        pattern=r'^[a-z0-9][a-z0-9-]*$',
        description="Unique extension identifier (lowercase, hyphens ok)",
    )
    version: str = Field(description="Semantic version string (e.g. '0.1.0')")
    description: str = Field(description="Human-readable description of the extension")
    author: Optional[str] = Field(default=None, description="Extension author name or org")
    hooks: list[str] = Field(
        default_factory=list,
        description="Event types this extension listens to (e.g. ['memory_stored', 'session_start'])",
    )


@runtime_checkable
class NeuralscapeExtension(Protocol):
    """Protocol that all NeuralScape extensions must implement.

    Extensions provide a manifest, lifecycle hooks (startup/shutdown),
    event handlers, and optional API routes.
    """

    manifest: ExtensionManifest

    async def startup(self) -> None:
        """Called when NeuralScape starts up.

        Use this for initializing resources, connections, or state
        that the extension needs throughout its lifetime.
        """
        ...

    async def shutdown(self) -> None:
        """Called when NeuralScape shuts down.

        Use this for cleaning up resources, closing connections,
        or flushing buffers.
        """
        ...

    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
        """Handle an event from NeuralScape core or other extensions.

        Only called for event types listed in manifest.hooks.

        Args:
            event_type: The type of event (e.g. 'memory_stored').
            payload: Event-specific data as a dictionary.

        Returns:
            Optional dict with any response data, or None.
        """
        ...

    def get_routes(self) -> Optional[APIRouter]:
        """Return an APIRouter to mount at /v1/extensions/<name>/.

        Returns:
            An APIRouter with the extension's HTTP endpoints, or None
            if the extension doesn't expose any routes.
        """
        ...
