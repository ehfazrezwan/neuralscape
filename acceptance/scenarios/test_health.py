"""Health endpoints answer without auth and report no dead backend."""

from __future__ import annotations

from nsaccept import expect_check_fails, negative_control, proves


def check_health_body(body: dict) -> None:
    assert body.get("status") in ("ok", "degraded"), f"status={body.get('status')!r}"
    checks = body.get("checks") or {}
    assert checks, "no per-backend checks in /health body"
    dead = {k: v for k, v in checks.items() if v == "unreachable"}
    assert not dead, f"unreachable backends: {dead}"
    # vector_store/graph_store initialise lazily (not_initialized is fine);
    # the queue backend must be connected.
    assert checks.get("redis") == "ok", f"redis={checks.get('redis')!r}"


@proves("ops.health-endpoints")
def test_live_probe_needs_no_auth(anon):
    r = anon.get("/health/live")
    assert r.status_code == 200, f"/health/live -> {r.status_code}"
    assert r.json().get("status") == "alive"


@proves("ops.health-endpoints")
def test_health_reports_backends_without_auth(anon):
    r = anon.get("/health")
    assert r.status_code == 200, f"/health -> {r.status_code}: {r.text[:200]}"
    check_health_body(r.json())


@proves("ops.health-endpoints")
@negative_control
def test_health_check_rejects_a_dead_backend():
    broken = {"status": "unhealthy", "checks": {"redis": "unreachable", "vector_store": "unreachable"}}
    expect_check_fails(check_health_body, broken)
