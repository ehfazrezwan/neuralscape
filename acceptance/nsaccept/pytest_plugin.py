"""pytest plugin: profile selection, @needs skips, results JSON, scenario listing.

Load with ``-p nsaccept.pytest_plugin`` (run.sh does) with acceptance/ on PYTHONPATH.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

import pytest

import hashlib

import httpx

from . import caps, guard, runctx
from .decor import DEFAULT_PROFILES
from .fixtures import a, anon, b, bad, ctx, mcp_a, mcp_b  # noqa: F401  (registered as fixtures)
from .tokens import scrub

ACCEPTANCE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("NS_REPO_ROOT") or ACCEPTANCE_DIR.parent)


# ── options / markers ───────────────────────────────────────────
def pytest_addoption(parser):
    g = parser.getgroup("nsaccept", "NeuralScape acceptance harness")
    g.addoption("--ns-base-url", default=os.environ.get("NS_BASE_URL"), help="stack base URL")
    g.addoption("--ns-profile", default="default", help="active profile name")
    g.addoption("--ns-out", default=None, help="results JSON path")
    g.addoption("--ns-compose-project", default=None, help="compose project for token minting")
    g.addoption(
        "--ns-list-out",
        default=None,
        help="write the scenario inventory JSON here and run nothing (no stack contact)",
    )


def pytest_configure(config):
    for name, doc in (
        ("ns_proves", "atlas node ids this scenario is evidence for"),
        ("ns_profiles", "profiles this scenario runs under (default: default)"),
        ("ns_needs", "capabilities required; scenario skips when absent"),
        ("ns_negative_control", "scenario is a negative control"),
    ):
        config.addinivalue_line("markers", f"{name}: {doc}")
    config._ns_state = {}
    config._ns_started = _dt.datetime.now(_dt.timezone.utc)
    config._ns_ctx = None
    config._ns_stack_commit = None


# ── helpers ─────────────────────────────────────────────────────
def _marker_args(item, name: str) -> list[str]:
    out: list[str] = []
    for m in item.iter_markers(name):
        out.extend(str(x) for x in m.args)
    return out


def scenario_id(item) -> str:
    return f"{Path(str(item.path)).stem}::{item.name}"


def item_profiles(item) -> tuple[str, ...]:
    declared = _marker_args(item, "ns_profiles")
    return tuple(declared) if declared else DEFAULT_PROFILES


def _commit() -> str:
    env = os.environ.get("NS_COMMIT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _utc(ts: _dt.datetime) -> str:
    return ts.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── collection: listing, profile deselection, context ───────────
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    list_out = config.getoption("--ns-list-out")
    if list_out:
        inventory = [
            {
                "id": scenario_id(it),
                "proves": _marker_args(it, "ns_proves"),
                "profiles": list(item_profiles(it)),
                "negative_control": it.get_closest_marker("ns_negative_control") is not None,
                "needs": _marker_args(it, "ns_needs"),
            }
            for it in items
        ]
        Path(list_out).write_text(json.dumps(inventory, indent=2) + "\n")
        config.hook.pytest_deselected(items=list(items))
        items[:] = []
        return
    active = config.getoption("--ns-profile")
    keep, drop = [], []
    for it in items:
        (keep if active in item_profiles(it) else drop).append(it)
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep


def _profile_info(profile: str) -> tuple[dict, str | None]:
    path = ACCEPTANCE_DIR / "profiles" / f"{profile}.env"
    if not path.is_file():
        return {}, None
    raw = path.read_bytes()
    flags: dict[str, str] = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        flags[key.strip()] = value.strip()
    return flags, hashlib.sha256(raw).hexdigest()


def _probe_stack_commit(base_url: str) -> str | None:
    """The stack's own commit if it exposes one in /health (no field exists today)."""
    try:
        body = httpx.get(base_url.rstrip("/") + "/health", timeout=10).json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    for key in ("commit", "git_sha", "git_commit", "build_sha"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _same_commit(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return a.startswith(b) or b.startswith(a)


def pytest_collection_finish(session):
    config = session.config
    if config.getoption("--ns-list-out"):
        return
    # Only scenarios that use the stack need a target, a guard check and tokens.
    if not any("ctx" in getattr(it, "fixturenames", ()) for it in session.items):
        return
    base = config.getoption("--ns-base-url")
    if not base:
        pytest.exit("harness misconfiguration: no base URL (pass it to run.sh)", returncode=2)
    compose = config.getoption("--ns-compose-project") or os.environ.get("NS_COMPOSE_PROJECT")
    try:
        guard.check(base, compose)
    except guard.GuardError as exc:
        pytest.exit(f"harness misconfiguration: {exc}", returncode=2)
    declared = os.environ.get("NS_COMMIT", "").strip()
    stack = os.environ.get("NS_STACK_COMMIT", "").strip() or _probe_stack_commit(base)
    if declared and stack and not _same_commit(declared, stack):
        pytest.exit(
            f"harness misconfiguration: NS_COMMIT={declared[:12]} but the stack reports {stack[:40]}",
            returncode=2,
        )
    config._ns_stack_commit = stack
    try:
        config._ns_ctx = runctx.build(
            base, config.getoption("--ns-profile"), config.getoption("--ns-compose-project")
        )
    except runctx.ConfigError as exc:
        pytest.exit(f"harness misconfiguration: {scrub(str(exc))}", returncode=2)


# ── @needs ──────────────────────────────────────────────────────
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    for cap in _marker_args(item, "ns_needs"):
        reason = caps.missing_reason(cap)
        if reason:
            pytest.skip(reason)


# ── results recording ───────────────────────────────────────────
def _short(text: str, limit: int = 400) -> str:
    return scrub(" ".join(text.split()))[:limit]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    state = item.config._ns_state.setdefault(
        item.nodeid,
        {
            "id": scenario_id(item),
            "proves": _marker_args(item, "ns_proves"),
            "result": "pass",
            "seconds": 0.0,
            "detail": "",
            "negative_control": item.get_closest_marker("ns_negative_control") is not None,
        },
    )
    state["seconds"] += rep.duration
    if rep.passed:
        return
    if rep.skipped:
        reason = ""
        if isinstance(rep.longrepr, tuple) and len(rep.longrepr) == 3:
            reason = str(rep.longrepr[2]).removeprefix("Skipped: ")
        elif call.excinfo is not None:
            reason = str(call.excinfo.value)
        if state["result"] == "pass":
            state["result"], state["detail"] = "skip", _short(reason)
        return
    # failed
    exc = call.excinfo
    is_assertion = exc is not None and exc.errisinstance((AssertionError, pytest.fail.Exception))
    if rep.when == "call" and is_assertion:
        kind = "fail"
    else:
        kind = "error"  # setup/teardown trouble, connection errors, harness bugs
    if state["result"] in ("pass", "skip") or (state["result"] == "error" and kind == "fail"):
        state["result"] = kind
        state["detail"] = _short(exc.exconly() if exc is not None else str(rep.longrepr))


def _results_path(config, commit: str) -> Path:
    out = config.getoption("--ns-out")
    if out:
        return Path(out)
    profile = config.getoption("--ns-profile")
    return ACCEPTANCE_DIR / "results" / f"{profile}-{commit[:12]}.json"


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if config.getoption("--ns-list-out") or not config._ns_state:
        return
    commit = _commit()
    flags, sha = _profile_info(config.getoption("--ns-profile"))
    doc = {
        "commit": commit,
        "stack_commit": config._ns_stack_commit,
        "profile": config.getoption("--ns-profile"),
        "profile_flags": flags,
        "profile_sha256": sha,
        "base_url": config.getoption("--ns-base-url") or "",
        "started": _utc(config._ns_started),
        "finished": _utc(_dt.datetime.now(_dt.timezone.utc)),
        "scenarios": [
            {**s, "seconds": round(s["seconds"], 2)} for s in config._ns_state.values()
        ],
    }
    path = _results_path(config, commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
