"""Real-service observations for the preservation comparison: observe facts, judge nothing.

  python comp1_realservice.py --service-dir DIR --compose-project NAME --compose-file FILE

Starts the target tree's API (its own interpreter, its own code, a free loopback port) against the
disposable stores of one compose project, then observes:

* ``readiness``: what ``/health`` reports with every store up, and again with the vector store
  container stopped (the store is really unreachable, not a fake);
* ``job_keys``: whether the API, on a real queue, coalesces two writes that differ only in project
  (extraction) or only in category (raw write) onto one task id, plus the user-scope controls.

No worker runs and no provider is called: writes are only enqueued.  Prints one JSON object of
short, content-free facts per scenario on the last line.  Synthetic identities only.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _compose(project: str, file: str, *args: str) -> str:
    done = subprocess.run(
        ["docker", "compose", "-p", project, "-f", file, *args],
        capture_output=True, text=True, check=True, timeout=180,
    )
    return done.stdout.strip()


def _port(project: str, file: str, service: str, inner: int) -> str:
    return _compose(project, file, "port", service, str(inner)).splitlines()[-1]


_TOKENS: dict[str, str] = {}  # user id -> bearer token (governed profile only)


def _call(
    base: str, method: str, path: str, body: dict | None = None, headers: dict | None = None
) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    headers = {"content-type": "application/json", **(headers or {})}
    user = (body or {}).get("user_id") or ("comp1-reader" if "comp1-reader" in path else None)
    if user in _TOKENS:
        headers["Authorization"] = f"Bearer {_TOKENS[user]}"
    request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"{}")
        except ValueError:
            return error.code, {}


def _start_api(service_dir: Path, env: dict[str, str]) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    python = service_dir / ".venv" / "bin" / "python"
    process = subprocess.Popen(
        [str(python), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(service_dir), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if _call(base, "GET", "/health/live")[0] in (200, 404):
                return process, base
        except OSError:
            pass
        time.sleep(0.5)
    process.kill()
    raise RuntimeError("api did not start")


def _token(value: object) -> str:
    text = str(value).replace(" ", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in "_.:+-")[:80] or "empty"


def readiness(base: str, project: str, file: str) -> dict[str, str]:
    # Initialise the vector-store client the way an ordinary read does.
    _call(base, "GET", "/v1/memories?user_id=comp1-reader&limit=1")
    up_status, up = _call(base, "GET", "/health")
    _compose(project, file, "stop", "qdrant")
    try:
        time.sleep(2)
        down_status, down = _call(base, "GET", "/health")
    finally:
        _compose(project, file, "start", "qdrant")
    # Exactly the unit assertions: the stopped store must not be reported ok, overall or per check.
    return {
        "stores_up_status_ok": str(up_status == 200 and up.get("status") == "ok").lower(),
        "stores_up_vector_store_ok": str(up.get("checks", {}).get("vector_store") == "ok").lower(),
        "stopped_status_ok": str(down.get("status") == "ok").lower(),
        "stopped_vector_store_ok": str(down.get("checks", {}).get("vector_store") == "ok").lower(),
    }


_FIXED_WRITE_FIELDS: dict[str, str] = {}  # e.g. an explicit visibility under an enforcing policy


def job_keys(base: str) -> dict[str, str]:
    messages = [{"role": "user", "content": "Remember that deploys freeze on Fridays (comp1)."}]

    def store(**extra) -> str:
        status, body = _call(base, "POST", "/v1/memories", {"messages": messages, **_FIXED_WRITE_FIELDS, **extra})
        return body.get("task_id") or f"http-{status}"

    def raw(**extra) -> str:
        payload = {"content": "Deploys freeze on Fridays (comp1).", "user_id": "comp1-alice",
                   "category": "convention", "scope": "project", "project_id": "alpha",
                   **_FIXED_WRITE_FIELDS, **extra}
        status, body = _call(base, "POST", "/v1/memories/raw", payload)
        return body.get("task_id") or f"http-{status}"

    alpha = store(user_id="comp1-alice", project_id="alpha")
    raw_base = raw()
    return {
        "store_same_inputs_same_task": str(store(user_id="comp1-alice", project_id="alpha") == alpha).lower(),
        "store_other_user_other_task": str(store(user_id="comp1-bob", project_id="alpha") != alpha).lower(),
        "store_other_project_other_task": str(store(user_id="comp1-alice", project_id="beta") != alpha).lower(),
        "store_other_agent_other_task": str(store(user_id="comp1-alice", project_id="alpha", agent_id="agent-2") != alpha).lower(),
        "raw_same_inputs_same_task": str(raw() == raw_base).lower(),
        "raw_other_project_other_task": str(raw(project_id="beta") != raw_base).lower(),
        "raw_other_category_other_task": str(raw(category="tech_stack") != raw_base).lower(),
        "raw_other_tags_other_task": str(raw(tags=["release"]) != raw_base).lower(),
        "store_accepted": "true" if not alpha.startswith("http-") else _token(alpha),
        "raw_accepted": "true" if not raw_base.startswith("http-") else _token(raw_base),
    }


def idempotency_retry(base: str) -> dict[str, str]:
    """An identical write retried with one Idempotency-Key, and a different write under it."""

    payload = {"content": "Releases ship on Tuesdays (comp1).", "user_id": "comp1-alice",
               "category": "convention", "scope": "project", "project_id": "alpha", **_FIXED_WRITE_FIELDS}
    key = {"Idempotency-Key": "comp1-retry-1"}
    first_status, first = _call(base, "POST", "/v1/memories/raw", payload, key)
    again_status, again = _call(base, "POST", "/v1/memories/raw", payload, key)
    changed_status, _ = _call(base, "POST", "/v1/memories/raw", {**payload, "category": "tech_stack"}, key)
    return {
        "first_http": str(first_status),
        "retry_same_task": str(bool(first.get("task_id")) and first.get("task_id") == again.get("task_id")).lower(),
        "retry_http": str(again_status),
        "changed_payload_same_key_http": str(changed_status),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-dir", required=True)
    parser.add_argument("--compose-project", required=True)
    parser.add_argument("--compose-file", required=True)
    parser.add_argument("--extra-env", action="append", default=[], help="KEY=VALUE for the API")
    parser.add_argument("--tokens", default="", help="JSON file mapping user id to bearer token")
    parser.add_argument("--write-visibility", default="", help="send this visibility on every write")
    args = parser.parse_args()
    if args.write_visibility:
        _FIXED_WRITE_FIELDS["visibility"] = args.write_visibility
    if args.tokens:
        _TOKENS.update(json.loads(Path(args.tokens).read_text()))
    service_dir = Path(args.service_dir).resolve()
    project, file = args.compose_project, args.compose_file
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEURALSCAPE_", "NS_", "GOOGLE_"))}
    env.update({
        "NEO4J_URI": "bolt://" + _port(project, file, "neo4j", 7687),
        "NEO4J_PASSWORD": "comp1-disposable-pw",
        "REDIS_URL": "redis://" + _port(project, file, "redis", 6379),
        "QDRANT_URL": "http://" + _port(project, file, "qdrant", 6333),
        "GOOGLE_API_KEY": "comp1-no-provider",  # never called: no worker runs, nothing is extracted
        "QDRANT_COLLECTION": "comp1",
    })
    env.update(dict(item.split("=", 1) for item in args.extra_env))
    out: dict[str, dict] = {}
    process, base = _start_api(service_dir, env)
    try:
        for name, probe in (("job_keys", lambda: job_keys(base)),
                            ("idempotency_retry", lambda: idempotency_retry(base)),
                            ("readiness", lambda: readiness(base, project, file))):
            try:
                out[name] = {"facts": probe()}
            except Exception as error:  # noqa: BLE001 - recorded as an invalid attempt by the caller
                out[name] = {"error": f"{type(error).__name__}: {str(error)[:160]}"}
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
