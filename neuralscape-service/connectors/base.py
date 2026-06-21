"""Connector adapter interface.

Every adapter knows how to (1) enumerate a source's resources incrementally
(``list_resources`` with a cursor), (2) fetch a resource's text
(``fetch``), and (3) describe how to re-fetch it (``retrieval_handle``).
The base class assembles those into the ``source_ref`` descriptor that the
ingest pipeline stamps onto each produced memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ConnectorResource:
    """One fetchable item in a source (a Drive file, a Notion page, …)."""
    external_id: str
    title: str | None = None
    url: str | None = None
    # Source-side version marker (etag / last_edited_time). The sync worker
    # compares this against the last-seen revision to skip unchanged resources.
    revision: str | None = None
    # Adapter-specific extra data carried from list_resources → fetch.
    raw: dict = field(default_factory=dict)


class ConnectorAdapter(ABC):
    """Base class for data-layer connectors.

    Subclasses set ``connector_type`` and implement the three abstract methods.
    Adapters are async (network I/O); the sync worker awaits them.
    """

    connector_type: str = "base"

    def __init__(self, connector_id: str, credentials: dict, config: dict | None = None):
        self.connector_id = connector_id
        self.credentials = credentials or {}
        self.config = config or {}

    @abstractmethod
    async def list_resources(
        self, cursor: str | None = None
    ) -> tuple[list[ConnectorResource], str | None]:
        """Return (resources, next_cursor). ``next_cursor`` is None when done.

        Implementations should honor ``cursor`` for incremental sync so a
        re-run only sees new/changed resources where the source supports it.
        """
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, resource: ConnectorResource) -> str:
        """Return the full text content of ``resource``."""
        raise NotImplementedError

    @abstractmethod
    def retrieval_handle(self, resource: ConnectorResource) -> dict:
        """Return a structured re-fetch handle: ``{mcp_server, tool, args}``.

        This is what a consuming agent uses to pull the original source again.
        """
        raise NotImplementedError

    def source_descriptor(self, resource: ConnectorResource) -> dict:
        """Assemble the base ``source_ref`` descriptor for a resource.

        The ingest pipeline extends this per-chunk for passages; facts use it
        as-is. Returned as a plain dict (matches ``SourceDescriptor`` fields)
        ready to drop into Qdrant metadata.
        """
        return {
            "connector_id": self.connector_id,
            "connector_type": self.connector_type,
            "external_id": resource.external_id,
            "parent_id": resource.external_id,
            "url": resource.url,
            "title": resource.title,
            "revision": resource.revision,
            "retrieval": self.retrieval_handle(resource),
        }

    async def aclose(self) -> None:
        """Release any network clients. Override if the adapter holds one."""
        return None
