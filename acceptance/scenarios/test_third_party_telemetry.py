"""Third-party telemetry (the vendored mem0 and Graphiti PostHog clients) is OFF by default.

Both libraries ship with anonymous usage telemetry on. The service switches it off before
they import (``third_party_telemetry.py``) and reports the effective state two ways, which
are what this file reads through the running stack's real surfaces:

- ``GET /health`` (no auth): ``checks.third_party_telemetry`` is
  ``{"mem0": "off"|"on", "graphiti": "off"|"on"}`` and does not by itself change ``status``.
- one startup log line in the API and in every worker: ``third_party_telemetry mem0=off graphiti=off``
  (read with ``ctx.compose("logs", ...)``, so it needs ``--compose-project``; without one the
  log scenario is skipped with a reason, which is not evidence).

What this shows: the state the service itself reports for each library, decided the way the
library decides. It is not a packet capture; absence of outbound connections is not observed.
The profiles never switch telemetry on: that would send real data to a third party.
"""

from __future__ import annotations

import re

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.checks import precondition
from nsaccept.runctx import app_services

NODES = (
    "fork.mem0.telemetry",
    "fork.graphiti.telemetry",
    "vec.mem0-posthog-telemetry",
    "ops.vendored-posthog-telemetry",
    "archive.pr451",
)
EXPECTED = {"mem0": "off", "graphiti": "off"}
LINE_RE = re.compile(r"third_party_telemetry mem0=(\S+?)\s+graphiti=([^\s\"',}]+)")


# ── the real checks ──────────────────────────────────────────────


def check_health_telemetry(body: dict) -> None:
    checks = body.get("checks")
    assert isinstance(checks, dict), "no checks object in /health body"
    assert "third_party_telemetry" in checks, f"no third_party_telemetry entry in checks (have {sorted(checks)})"
    got = checks["third_party_telemetry"]
    assert got == EXPECTED, f"third_party_telemetry={got!r}, want {EXPECTED!r}"


def check_telemetry_does_not_degrade_status(body: dict) -> None:
    """``status`` follows the other checks only: ok exactly when every one of them is ok."""
    others = {k: v for k, v in (body.get("checks") or {}).items() if k != "third_party_telemetry"}
    want = "ok" if all(v == "ok" for v in others.values()) else "degraded"
    assert body.get("status") == want, (
        f"status={body.get('status')!r} but the other checks {others!r} give {want!r}: "
        "the telemetry entry must not move it"
    )


def last_telemetry_line(text: str):
    found = LINE_RE.findall(text or "")
    return found[-1] if found else None


def check_telemetry_log(service: str, text: str) -> None:
    last = last_telemetry_line(text)
    assert last is not None, f"{service}: no 'third_party_telemetry mem0=... graphiti=...' startup line in its logs"
    got = {"mem0": last[0], "graphiti": last[1]}
    assert got == EXPECTED, f"{service}: last startup line says {got!r}, want {EXPECTED!r}"


def check_all_services_log_off(logs: dict[str, str]) -> None:
    assert logs, "no services to check"
    for service, text in logs.items():
        check_telemetry_log(service, text)


# ── /health ──────────────────────────────────────────────────────


@proves(*NODES)
def test_health_reports_third_party_telemetry_off(anon):
    r = anon.get("/health")
    assert r.status_code == 200, f"/health -> {r.status_code}: {r.text[:200]}"
    check_health_telemetry(r.json())


@proves(*NODES)
def test_health_telemetry_entry_does_not_degrade_status(anon):
    r = anon.get("/health")
    assert r.status_code == 200, f"/health -> {r.status_code}"
    body = r.json()
    check_health_telemetry(body)
    check_telemetry_does_not_degrade_status(body)


@proves(*NODES)
@negative_control
def test_health_telemetry_check_fails_for_on_or_missing():
    good = {"status": "ok", "checks": {"redis": "ok", "third_party_telemetry": {"mem0": "off", "graphiti": "off"}}}
    precondition(check_health_telemetry, good, what="a correct /health body")
    for bad in (
        {"mem0": "on", "graphiti": "off"},
        {"mem0": "off", "graphiti": "on"},
        {"mem0": "on", "graphiti": "on"},
        {"mem0": "off"},
        "off",
    ):
        expect_check_fails(check_health_telemetry, {**good, "checks": {"redis": "ok", "third_party_telemetry": bad}})
    expect_check_fails(check_health_telemetry, {"status": "ok", "checks": {"redis": "ok"}})
    expect_check_fails(check_health_telemetry, {"status": "ok"})
    # status check: ok iff every other check is ok
    precondition(check_telemetry_does_not_degrade_status, good, what="status ok with every other check ok")
    precondition(check_telemetry_does_not_degrade_status, {"status": "degraded", "checks": {"redis": "unreachable", "third_party_telemetry": EXPECTED}}, what="degraded by a real backend")
    expect_check_fails(check_telemetry_does_not_degrade_status, {**good, "status": "degraded"})  # degraded by telemetry alone
    expect_check_fails(check_telemetry_does_not_degrade_status, {"status": "ok", "checks": {"redis": "unreachable", "third_party_telemetry": EXPECTED}})


# ── startup log lines ────────────────────────────────────────────


def service_logs(ctx) -> dict[str, str]:
    return {svc: (ctx.compose("logs", "--no-log-prefix", svc, check=False).stdout or "") for svc in app_services()}


@proves(*NODES)
@needs("compose")
def test_api_and_every_worker_log_third_party_telemetry_off(ctx):
    check_all_services_log_off(service_logs(ctx))


@proves(*NODES)
@negative_control
def test_telemetry_log_check_fails_for_on_or_no_line():
    ok = 'starting\n{"event": "third_party_telemetry mem0=off graphiti=off", "level": "info"}\nready\n'
    precondition(check_telemetry_log, "api", ok, what="correct log text")
    precondition(check_all_services_log_off, {"api": ok, "worker": "third_party_telemetry mem0=off graphiti=off\n"}, what="two correct services")
    # the LAST line decides: an old "off" line followed by "on" fails, "on" followed by "off" passes
    expect_check_fails(check_telemetry_log, "api", ok + "third_party_telemetry mem0=on graphiti=off\n")
    expect_check_fails(check_telemetry_log, "api", ok + "third_party_telemetry mem0=off graphiti=on\n")
    precondition(check_telemetry_log, "api", "third_party_telemetry mem0=on graphiti=on\n" + ok, what="restart fixed an old 'on'")
    expect_check_fails(check_telemetry_log, "api", "starting\nready\n")
    expect_check_fails(check_telemetry_log, "api", "")
    # a service with no line fails the all-services check and is named
    expect_check_fails(check_all_services_log_off, {"api": ok, "neuralscape-worker": "starting\n"})
    expect_check_fails(check_all_services_log_off, {})
