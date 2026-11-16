"""Tiny MCP Streamable HTTP server used by test_connector_ssrf_guard.py.

Run as ``<python> <this file> <port>``; listens on 127.0.0.1 only and exposes
one text resource.
"""

import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ns-http-fixture", host="127.0.0.1", port=int(sys.argv[1]))


@mcp.resource("fixture://gamma", name="gamma", mime_type="text/plain")
def gamma() -> str:
    return "gamma resource body: http connector works"


if __name__ == "__main__":
    mcp.run("streamable-http")
