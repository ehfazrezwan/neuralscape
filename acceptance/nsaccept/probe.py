"""Route-probe helper: does a route exist and answer sanely?

A probe sends a request that cannot do harm (invalid body, random ids, dry-run
flags) and classifies the answer. "Present" means the router matched the path
and method: not 405, not FastAPI's own "Not Found" 404, and not a server error.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

# FastAPI's body for a path that no route matches. Resource-level 404s carry a
# specific detail ("Memory x not found") instead.
UNROUTED_DETAIL = "Not Found"


@dataclass
class ProbeResult:
    method: str
    path: str
    status: int
    detail: str
    present: bool
    reason: str

    def line(self) -> str:
        return f"{self.method} {self.path} -> {self.status} ({self.reason})"


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:120]
    if isinstance(body, dict):
        return str(body.get("detail", ""))[:200]
    return ""


def classify(method: str, path: str, resp: httpx.Response) -> ProbeResult:
    status, detail = resp.status_code, _detail(resp)
    if status == 405:
        return ProbeResult(method, path, status, detail, False, "method not allowed: route missing for this verb")
    if status == 404 and detail.strip() == UNROUTED_DETAIL:
        return ProbeResult(method, path, status, detail, False, "route not mounted")
    if status == 501:  # optional extra not installed (code-graph): the route is there and says so
        return ProbeResult(method, path, status, detail, True, "optional extra absent (501)")
    if status == 503 and "disabled" in detail.lower():
        return ProbeResult(method, path, status, detail, True, "feature disabled (503)")
    if status >= 500:
        return ProbeResult(method, path, status, detail, False, f"server error: {detail}")
    return ProbeResult(method, path, status, detail, True, "ok")


def probe_route(client, method: str, path: str, **kw) -> ProbeResult:
    """Send one request through an NSClient and classify the answer."""
    return classify(method, path, client.request(method, path, **kw))
