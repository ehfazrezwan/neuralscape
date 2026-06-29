"""Map a connector_type string to its adapter class + build instances.

Adapters are imported lazily inside :func:`get_adapter_class` so importing the
registry (e.g. from the API layer) doesn't drag in optional adapter deps
(httpx, the mcp client) until a connector of that type is actually used.
"""

from __future__ import annotations

from connectors.base import ConnectorAdapter


def get_adapter_class(connector_type: str) -> type[ConnectorAdapter]:
    """Return the adapter class for ``connector_type``. Raises ValueError if unknown."""
    if connector_type == "notion":
        from connectors.adapters.notion import NotionAdapter

        return NotionAdapter
    if connector_type == "google_drive":
        from connectors.adapters.gdrive import GoogleDriveAdapter

        return GoogleDriveAdapter
    if connector_type in ("mcp", "generic_rest"):
        from connectors.adapters.mcp_generic import MCPConnectorAdapter

        # generic_rest is handled by the same generic adapter family today;
        # both are configured via the instance's `config` block.
        if connector_type == "generic_rest":
            from connectors.adapters.generic_rest import GenericRestAdapter

            return GenericRestAdapter
        return MCPConnectorAdapter
    raise ValueError(f"Unknown connector_type: {connector_type}")


def build_adapter(record: dict) -> ConnectorAdapter:
    """Instantiate an adapter from a vault record (with decrypted credentials)."""
    cls = get_adapter_class(record["connector_type"])
    return cls(
        connector_id=record["connector_id"],
        credentials=record.get("credentials", {}),
        config=record.get("config", {}),
    )
