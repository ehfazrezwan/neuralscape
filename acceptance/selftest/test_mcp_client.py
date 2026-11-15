import pytest

from nsaccept.mcp import MCPClient, McpError, parse_rpc_body


@pytest.mark.parametrize("mode", ["json", "sse", "session"])
def test_full_handshake_and_tool_calls(stub, mode):
    state, url = stub
    state.mcp_mode = mode
    c = MCPClient(url, "good-token")
    assert c.initialize()["serverInfo"]["name"] == "stub"
    assert [t["name"] for t in c.list_tools()] == ["remember"]
    res = c.call_tool("list_memories", {"limit": 1})
    assert MCPClient.result_text(res) == "[]"
    # Accept advertises both transports on every request
    assert all("text/event-stream" in h["accept"] and "application/json" in h["accept"] for h in state.seen_headers)
    if mode == "session":
        assert c.session_id == "sess-1"
        assert state.seen_headers[-1]["mcp-session-id"] == "sess-1"
    c.close()


def test_unauthenticated_is_http_error(stub):
    _, url = stub
    c = MCPClient(url)
    with pytest.raises(McpError) as ei:
        c.initialize()
    assert ei.value.status == 401
    assert c.post_raw({"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401


def test_parse_rpc_body_picks_matching_id_and_skips_notifications():
    sse = 'data: {"jsonrpc":"2.0","method":"x"}\n\ndata: {"jsonrpc":"2.0","id":7,"result":{"a":1}}\n\n'
    assert parse_rpc_body("text/event-stream", sse, 7)["result"] == {"a": 1}
    assert parse_rpc_body("text/event-stream", sse, 8) is None
    assert parse_rpc_body("application/json", '[{"id":1,"result":{}}]', 1)["result"] == {}
    assert parse_rpc_body("application/json", "", 1) is None
