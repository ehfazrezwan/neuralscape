"""Data-layer connectors: pull content from external systems into memory.

Neuralscape acts as an MCP/API *client* here (it is an MCP server everywhere
else). A connector instance is a configured source (a Notion workspace, a
Google Drive account, a remote MCP server) whose credentials live encrypted in
the :class:`~connectors.vault.ConnectorVault`. The sync worker walks each
connector's resources and feeds them through the ingest pipeline, stamping a
``source_ref`` on every produced memory.
"""

from connectors.base import ConnectorAdapter, ConnectorResource
from connectors.registry import build_adapter, get_adapter_class
from connectors.vault import ConnectorVault

__all__ = [
    "ConnectorAdapter",
    "ConnectorResource",
    "build_adapter",
    "get_adapter_class",
    "ConnectorVault",
]
