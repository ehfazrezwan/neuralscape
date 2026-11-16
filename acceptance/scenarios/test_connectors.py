"""Data-layer connectors with the flag ON (profile ``flags-on``): a REST connector pointed at
the ``nsc-acc-sink`` sidecar's fixture JSON, the encrypted credential vault, per-owner
isolation, incremental sync, and the MCP stdio allowlist.

The sink serves two fixture items and logs every request it receives, so the credential the
vault decrypts for the sync is observable on the wire (it is a unique marker, not a secret).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, profiles, proves
from nsaccept.client import NSClient
from nsaccept.sidecar import CONNECTOR_ITEMS_PATH, SINK_NAME, PORT, connector_marker

pytestmark = [profiles("flags-on"), needs("compose")]

SYNC_WAIT = 300
REDIS_SERVICE = os.environ.get("NS_REDIS_SERVICE", "redis")


# ── the real checks ──────────────────────────────────────────────


def check_no_secret(body, secret: str) -> None:
    assert secret not in json.dumps(body, default=str), "the stored credential came back in an API response"


def check_recalled_from_connector(client: NSClient, marker: str, project_id: str, connector_id: str, external_id: str) -> dict:
    hits = [h for h in client.search(f"{marker} tide table quay pilot hut", project_id=project_id, limit=20) if marker in h.get("memory", "")]
    assert hits, f"{marker} not recalled by {client.user_id}"
    ref = hits[0].get("source_ref") or {}
    assert ref.get("connector_id") == connector_id, f"source_ref.connector_id {ref.get('connector_id')!r}"
    assert ref.get("external_id") == external_id, f"source_ref.external_id {ref.get('external_id')!r}"
    return hits[0]


def check_stdio_refused(client: NSClient, body: dict) -> None:
    r = client.post("/v1/connectors", json=body)
    assert r.status_code == 422, f"stdio connector {body['config']} -> {r.status_code} (want 422): {r.text[:160]}"


# ── fixture: register + sync once, assert many ───────────────────


@dataclass
class Conn:
    connector_id: str
    credential: str
    created: dict
    first_sync: dict
    second_sync: dict
    error: str = ""


def _record(ctx, conn_id: str, credential: str, **over) -> dict:
    rec = {
        "connector_id": conn_id,
        "connector_type": "generic_rest",
        "name": "acceptance REST connector",
        "credentials": {"bearer": credential},
        "config": {
            "list_url": f"http://{SINK_NAME}:{PORT}{CONNECTOR_ITEMS_PATH}",
            "items_path": "items", "id_field": "id", "title_field": "title",
            "revision_field": "rev", "content_field": "body",
        },
        "default_category": "domain_knowledge",
        "default_scope": "project",
        "default_project_id": ctx.project_id,
        "default_visibility": "private",
    }
    rec.update(over)
    return rec


def _sync(client: NSClient, conn_id: str) -> dict:
    r = client.post(f"/v1/connectors/{conn_id}/sync")
    assert r.status_code == 202, f"POST /v1/connectors/{conn_id}/sync -> {r.status_code}: {r.text[:200]}"
    st = client.poll_task(r.json()["task_id"], timeout=SYNC_WAIT)
    assert st.get("status") == "completed", f"connector sync ended as {st}"
    return st["result"]


@pytest.fixture(scope="module")
def conn(ctx, a, sink) -> Conn:
    conn_id = f"acc-{ctx.run_id}-rest"
    credential = f"zq-{ctx.run_id}-vaultcred"
    c = Conn(conn_id, credential, {}, {}, {})
    try:
        r = a.post("/v1/connectors", json=_record(ctx, conn_id, credential))
        assert r.status_code == 201, f"POST /v1/connectors -> {r.status_code}: {r.text[:200]}"
        c.created = r.json()
        c.first_sync = _sync(a, conn_id)
        c.second_sync = _sync(a, conn_id)
    except AssertionError as exc:
        c.error = str(exc)
    return c


def _ready(c: Conn) -> Conn:
    assert not c.error, c.error
    return c


# ── register / vault ─────────────────────────────────────────────


@proves("ingest.connector-rest-api", "ingest.connector-vault", "auth.connector-vault")
def test_registering_stores_the_credential_encrypted_and_never_returns_it(ctx, a, conn: Conn):
    _ready(conn)
    check_no_secret(conn.created, conn.credential)
    assert conn.created["connector_id"] == conn.connector_id and conn.created["has_credentials"] is True, f"created: {conn.created}"
    assert conn.created["owner_user_id"] == ctx.user_a, f"owner {conn.created.get('owner_user_id')!r}"
    one = a.get(f"/v1/connectors/{conn.connector_id}")
    assert one.status_code == 200, f"GET connector -> {one.status_code}"
    check_no_secret(one.json(), conn.credential)
    listing = a.get("/v1/connectors").json()
    assert any(c["connector_id"] == conn.connector_id for c in listing["connectors"]), "connector not in the caller's list"
    check_no_secret(listing, conn.credential)
    # at rest, in Redis: ciphertext only
    raw = ctx.compose("exec", "-T", REDIS_SERVICE, "redis-cli", "GET", f"ns:connector:{conn.connector_id}").stdout
    assert "credentials_enc" in raw, f"vault record has no encrypted blob: {raw[:120]!r}"
    assert conn.credential not in raw, "the credential is stored in clear in Redis"


@proves("ingest.connector-vault", "auth.connector-vault")
@negative_control
def test_secret_check_fails_when_the_credential_is_in_the_body(conn: Conn):
    _ready(conn)
    expect_check_fails(check_no_secret, {"credentials": {"bearer": conn.credential}}, conn.credential)


# ── sync ─────────────────────────────────────────────────────────


@proves("ingest.connector-sync", "ingest.connector-rest-api", "ingest.connector-adapters", "ingest.connector-vault")
def test_a_sync_imports_the_stub_sources_items_with_the_decrypted_credential(ctx, a, sink, conn: Conn):
    _ready(conn)
    assert conn.first_sync.get("synced") == 2 and conn.first_sync.get("skipped") == 0, f"first sync: {conn.first_sync}"
    pulls = [r for r in sink.requests("GET", CONNECTOR_ITEMS_PATH)]
    assert pulls, "the sink never saw the connector's list request"
    assert pulls[0]["authorization"] == f"Bearer {conn.credential}", "the sync did not send the vault's decrypted bearer credential"
    for n in (1, 2):
        eventually(
            lambda n=n: check_recalled_from_connector(a, connector_marker(ctx.run_id, n), ctx.project_id, conn.connector_id, f"item-{n}"),
            timeout=90, interval=3, what=f"connector item {n} recallable",
        )


@proves("ingest.connector-sync")
@negative_control
def test_recall_check_fails_for_user_b_and_for_the_wrong_item(ctx, b, conn: Conn):
    _ready(conn)
    expect_check_fails(check_recalled_from_connector, b, connector_marker(ctx.run_id, 1), ctx.project_id, conn.connector_id, "item-1")


@proves("ingest.connector-sync")
def test_a_second_sync_of_unchanged_revisions_ingests_nothing(conn: Conn, sink):
    _ready(conn)
    assert conn.second_sync.get("synced") == 0 and conn.second_sync.get("skipped") == 2, f"second sync: {conn.second_sync}"
    assert len(sink.requests("GET", CONNECTOR_ITEMS_PATH)) >= 2, "the second sync did not poll the source"


# ── ownership ────────────────────────────────────────────────────


@proves("ingest.connector-rest-api", "auth.connector-vault")
def test_another_user_cannot_see_sync_replace_or_delete_a_connector(ctx, a, b, conn: Conn):
    _ready(conn)
    cid = conn.connector_id
    assert b.get(f"/v1/connectors/{cid}").status_code == 404, "B can read A's connector"
    assert all(c["connector_id"] != cid for c in b.get("/v1/connectors").json()["connectors"]), "A's connector is in B's list"
    assert b.post(f"/v1/connectors/{cid}/sync").status_code == 404, "B can sync A's connector"
    takeover = b.post("/v1/connectors", json=_record(ctx, cid, "zq-b-takeover"))
    assert takeover.status_code == 404, f"B re-registering A's id -> {takeover.status_code} (want 404)"
    assert b.delete(f"/v1/connectors/{cid}").status_code == 404, "B can delete A's connector"
    still = a.get(f"/v1/connectors/{cid}")
    assert still.status_code == 200 and still.json()["owner_user_id"] == ctx.user_a, f"A's connector after B's attempts: {still.status_code} {still.text[:120]}"
    assert not [h for h in b.search(connector_marker(ctx.run_id, 1), project_id=ctx.project_id) if connector_marker(ctx.run_id, 1) in h.get("memory", "")], "B recalls A's private imports"


@proves("ingest.connector-rest-api")
def test_delete_removes_the_connector_but_not_what_it_imported(ctx, a, conn: Conn):
    _ready(conn)
    cid = f"acc-{ctx.run_id}-del"
    assert a.post("/v1/connectors", json=_record(ctx, cid, f"zq-{ctx.run_id}-delcred")).status_code == 201
    r = a.delete(f"/v1/connectors/{cid}")
    assert r.status_code == 200 and r.json().get("deleted") is True, f"DELETE -> {r.status_code}: {r.text[:120]}"
    assert a.get(f"/v1/connectors/{cid}").status_code == 404, "deleted connector still readable"
    assert a.post(f"/v1/connectors/{cid}/sync").status_code == 404, "deleted connector still syncs"
    # the earlier sync's memories (a different connector id) are untouched by deleting this one
    check_recalled_from_connector(a, connector_marker(ctx.run_id, 1), ctx.project_id, conn.connector_id, "item-1")
    bad = a.post("/v1/connectors", json=_record(ctx, f"acc-{ctx.run_id}-bad", "x", connector_type="no-such-type"))
    assert bad.status_code == 422, f"unknown connector_type -> {bad.status_code} (want 422)"


# ── MCP stdio allowlist ──────────────────────────────────────────


def _mcp(ctx, conn_id: str, config: dict) -> dict:
    return {
        "connector_id": conn_id, "connector_type": "mcp", "credentials": {}, "config": config,
        "default_category": "domain_knowledge", "default_scope": "global",
    }


@proves("auth.connector-mcp-stdio")
def test_only_an_allowlisted_stdio_command_may_be_registered(ctx, a):
    # refused: a command the operator never approved, extra args on an approved one, env, no command
    check_stdio_refused(a, _mcp(ctx, f"acc-{ctx.run_id}-sh", {"transport": "stdio", "command": "sh", "args": ["-c", "id"]}))
    check_stdio_refused(a, _mcp(ctx, f"acc-{ctx.run_id}-extra", {"transport": "stdio", "command": "echo", "args": ["acc-allowed-stdio", "--more"]}))
    check_stdio_refused(a, _mcp(ctx, f"acc-{ctx.run_id}-env", {"transport": "stdio", "command": "echo", "args": ["acc-allowed-stdio"], "env": {"LD_PRELOAD": "/x"}}))
    check_stdio_refused(a, _mcp(ctx, f"acc-{ctx.run_id}-nocmd", {"transport": "stdio"}))
    assert a.get(f"/v1/connectors/acc-{ctx.run_id}-sh").status_code == 404, "a refused connector was stored anyway"
    # allowed: exactly the line in the profile's CONNECTOR_STDIO_COMMANDS (registered only, never synced)
    ok_id = f"acc-{ctx.run_id}-stdio"
    allowed = a.post("/v1/connectors", json=_mcp(ctx, ok_id, {"transport": "stdio", "command": "echo", "args": ["acc-allowed-stdio"]}))
    assert allowed.status_code == 201, f"allowlisted stdio command -> {allowed.status_code}: {allowed.text[:160]}"
    assert a.delete(f"/v1/connectors/{ok_id}").status_code == 200


@proves("auth.connector-mcp-stdio")
@negative_control
def test_stdio_refusal_check_fails_for_the_allowlisted_command(ctx, a):
    ok_id = f"acc-{ctx.run_id}-stdio0"
    expect_check_fails(check_stdio_refused, a, _mcp(ctx, ok_id, {"transport": "stdio", "command": "echo", "args": ["acc-allowed-stdio"]}))
    a.delete(f"/v1/connectors/{ok_id}")


# ── destination policy (SSRF guard) ──────────────────────────────


def check_destination_refused(client: NSClient, body: dict) -> None:
    r = client.post("/v1/connectors", json=body)
    assert r.status_code == 422, f"{body['config']} -> {r.status_code}: {r.text[:160]}"
    assert client.get(f"/v1/connectors/{body['connector_id']}").status_code == 404, "a refused connector was stored anyway"


def _rest_to(ctx, conn_id: str, list_url: str) -> dict:
    return {
        "connector_id": conn_id, "connector_type": "generic_rest", "credentials": {"bearer": f"zq-{ctx.run_id}-ssrf"},
        "config": {"list_url": list_url, "items_path": "items"},
        "default_category": "domain_knowledge", "default_scope": "global",
    }


@proves("ingest.connector-rest-api", "auth.connector-mcp-stdio")
def test_connectors_cannot_target_internal_or_metadata_addresses(ctx, a):
    # An internal host that the profile does NOT allowlist (same host, other port), loopback,
    # the cloud metadata address, an IPv4-mapped loopback, a decimal-spelled loopback.
    for n, url in enumerate((
        f"http://{SINK_NAME}:9999/items", "http://127.0.0.1:8199/health", "https://127.0.0.1/items",
        "https://169.254.169.254/latest/meta-data/", "https://[::ffff:127.0.0.1]/items", "https://2130706433/items",
    )):
        check_destination_refused(a, _rest_to(ctx, f"acc-{ctx.run_id}-ssrf{n}", url))
    check_destination_refused(a, _mcp(ctx, f"acc-{ctx.run_id}-ssrfm", {"url": f"https://{SINK_NAME}:9999/mcp"}))


@proves("ingest.connector-rest-api", "auth.connector-mcp-stdio")
@negative_control
def test_destination_refusal_check_fails_for_the_allowlisted_sink(ctx, a):
    ok_id = f"acc-{ctx.run_id}-ssrf0"
    expect_check_fails(check_destination_refused, a, _rest_to(ctx, ok_id, f"http://{SINK_NAME}:{PORT}{CONNECTOR_ITEMS_PATH}"))
    a.delete(f"/v1/connectors/{ok_id}")
