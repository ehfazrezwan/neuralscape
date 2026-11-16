"""Shared helpers for the Phase 2 story scenarios (``test_phase2_*``).

Everything talks to the stack through REST only. Nothing here asserts at import or fixture time except
the ``check_*`` functions, which are the REAL checks a scenario and its negative control both call.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any

from nsaccept import tokens
from nsaccept.client import NSClient
from nsaccept.helpers import memory_ids_in

# The reasons carried by @expected_red: a closed set of neutral slice names (see nsaccept/redmark.py).
RED_DURABLE_WRITES = "needs Phase 2 durable write cutover"
RED_MEMBERSHIP = "needs Phase 2 membership and onboarding"
RED_ONBOARDING_POLICY = "needs Phase 2 onboarding policy"
RED_TENANT_CELLS = "needs Phase 2 tenant cells"
RED_PORTABILITY = "needs Phase 2 portable export"

DENIED = (403, 404)  # the policy layer answers 403 for a named denial and 404 for an existence-safe one


# ── identities ───────────────────────────────────────────────────


def mint_identity(ctx, user_id: str) -> str:
    """A valid token for ``user_id`` (a user the authority store has never been told about)."""
    secret = os.environ.get("NS_TOKEN_SECRET", "").strip()
    if secret:
        return tokens.mint_local(user_id, secret)
    if ctx.compose_project:
        return tokens.mint_via_compose(ctx.compose_project, user_id, os.environ.get("NS_API_SERVICE", "neuralscape"))
    raise AssertionError("cannot mint an extra identity: run with --compose-project or NS_TOKEN_SECRET")


def client_for(ctx, user_id: str) -> NSClient:
    return NSClient(ctx.base_url, mint_identity(ctx, user_id), user_id)


# ── records ──────────────────────────────────────────────────────


def receipt_of(status: dict) -> dict | None:
    result = status.get("result")
    receipt = result.get("receipt") if isinstance(result, dict) else None
    return receipt if isinstance(receipt, dict) else None


def memory_id_of(status: dict) -> str:
    """The memory a completed write made: from ``result.memories`` today, from the receipt outputs once durable."""
    ids = memory_ids_in(status)
    if ids:
        return ids[0]
    receipt = receipt_of(status)
    for out in (receipt or {}).get("outputs", []):
        if isinstance(out, dict) and out.get("kind") == "memory" and out.get("id"):
            return out["id"]
    return ""


def revision_of(status: dict) -> int | None:
    receipt = receipt_of(status)
    for out in (receipt or {}).get("outputs", []):
        if isinstance(out, dict) and out.get("kind") == "memory" and isinstance(out.get("revision"), int):
            return out["revision"]
    return None


def check_recalled(client: NSClient, marker: str, query: str) -> dict:
    hits = client.search(query, limit=20)
    for hit in hits:
        if marker in hit.get("memory", ""):
            return hit
    raise AssertionError(f"{marker} not among {len(hits)} search results")


def check_not_recalled(client: NSClient, marker: str, query: str) -> None:
    hits = client.search(query, limit=20)
    assert not any(marker in h.get("memory", "") for h in hits), f"{marker} was recalled"


def check_readable_by_id(client: NSClient, memory_id: str, marker: str) -> dict:
    r = client.get(f"/v1/memories/{memory_id}")
    assert r.status_code == 200, f"GET /v1/memories/<id> -> {r.status_code}: {r.text[:120]}"
    row = r.json()
    assert marker in row.get("memory", ""), f"row does not carry {marker}"
    return row


def check_blind_by_id(client: NSClient, memory_id: str) -> None:
    r = client.get(f"/v1/memories/{memory_id}")
    assert r.status_code in DENIED, f"GET /v1/memories/<id> -> {r.status_code} (want 403/404)"


# ── export ───────────────────────────────────────────────────────


def okf_export(client: NSClient) -> bytes:
    r = client.get("/v1/export/okf")
    assert r.status_code == 200, f"GET /v1/export/okf -> {r.status_code}: {r.text[:160]}"
    assert r.content[:2] == b"PK", "export is not a zip archive"
    return r.content


def bundle_text(data: bytes) -> str:
    """Every member of an exported zip, decoded and concatenated (for substring checks)."""
    out: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            out.append(zf.read(name).decode("utf-8", "replace"))
    return "\n".join(out)


def check_export_carries(client: NSClient, marker: str) -> bytes:
    data = okf_export(client)
    assert marker in bundle_text(data), f"{marker} is not in the caller's export"
    return data


# ── what one fixture step recorded ───────────────────────────────


@dataclass
class Step:
    """One HTTP step of a story, recorded without asserting (a fixture error would read as ``error``)."""

    name: str
    status: int = 0
    body: Any = None
    task_id: str = ""
    final: dict = field(default_factory=dict)  # the polled task status, when the step queued one
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and 200 <= self.status < 300

    def describe(self) -> str:
        return f"{self.name}: {self.error or self.status} {str(self.body)[:160]}"


def call(name: str, client: NSClient, method: str, path: str, *, poll: bool = True, **kw: Any) -> Step:
    """Send one request, record it, and poll a queued task to its end."""
    step = Step(name)
    try:
        r = client.request(method, path, **kw)
        step.status = r.status_code
        try:
            step.body = r.json()
        except ValueError:
            step.body = r.text[:200]
        if isinstance(step.body, dict):
            step.task_id = str(step.body.get("task_id") or "")
        if poll and step.task_id and r.status_code in (200, 202):
            step.final = client.poll_task(step.task_id, timeout=120)
    except AssertionError as exc:
        step.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - connection trouble is recorded, the scenario reports it
        step.error = f"{type(exc).__name__}: {exc}"[:200]
    return step
