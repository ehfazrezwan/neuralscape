"""Tiny MCP stdio server used by test_connector_stdio_allowlist.py.

Exposes two text resources. Run as ``<python> <this file>``.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ns-stdio-fixture")


@mcp.resource("fixture://alpha", name="alpha", mime_type="text/plain")
def alpha() -> str:
    return "alpha resource body: stdio connector works"


@mcp.resource("fixture://beta", name="beta", mime_type="text/plain")
def beta() -> str:
    return "beta resource body"


if __name__ == "__main__":
    mcp.run("stdio")
