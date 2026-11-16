"""Small in-network client for lifecycle journeys (standard library only).

Commands: ``write N TAG`` (raw writes), ``drain`` (wait for empty queues), ``fingerprint``
(count and digest of every stored memory the API lists), ``search`` (hit count for a query).
Prints one JSON object.
"""

import hashlib
import http.client
import json
import sys
import time
import urllib.parse

BASE = "http://api:8199"
USER = "journey-user"


def call(method, path, body=None):
    u = urllib.parse.urlparse(BASE)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
    conn.request(method, path, body=None if body is None else json.dumps(body),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    try:
        return r.status, json.loads(data) if data else None
    except json.JSONDecodeError:
        return r.status, None


def drain(timeout=180):
    import redis

    c = redis.Redis.from_url("redis://redis:6379", socket_timeout=5)
    queues = ("neuralscape:queue", "neuralscape:graph", "neuralscape:ingest", "neuralscape:queue:ledger")
    start, quiet = time.monotonic(), None
    while time.monotonic() - start < timeout:
        pending = sum(c.zcard(q) for q in queues) + sum(
            1 for k in c.scan_iter(match="arq:in-progress:*") if not k.decode().startswith("arq:in-progress:cron:"))
        if pending == 0:
            quiet = quiet or time.monotonic()
            if time.monotonic() - quiet >= 2:
                return {"drained": True, "seconds": round(time.monotonic() - start, 1)}
        else:
            quiet = None
        time.sleep(0.5)
    return {"drained": False, "seconds": round(time.monotonic() - start, 1)}


def main():
    cmd = sys.argv[1]
    out = {}
    if cmd == "write":
        n, tag = int(sys.argv[2]), sys.argv[3]
        codes = {}
        for i in range(n):
            s, _ = call("POST", "/v1/memories/raw", {"content": f"Journey fact {tag}-{i}: {hashlib.sha256(f'{tag}{i}'.encode()).hexdigest()[:20]}",
                                                     "category": "preference", "user_id": USER, "visibility": "private"})
            codes[s] = codes.get(s, 0) + 1
        out = {"status_codes": codes, **drain()}
    elif cmd == "writenodrain":
        n, tag = int(sys.argv[2]), sys.argv[3]
        codes = {}
        for i in range(n):
            st, _ = call("POST", "/v1/memories/raw", {"content": f"Journey fact {tag}-{i}: {hashlib.sha256(f'{tag}{i}'.encode()).hexdigest()[:20]}",
                                                      "category": "preference", "user_id": USER, "visibility": "private"})
            codes[st] = codes.get(st, 0) + 1
        out = {"status_codes": codes}
    elif cmd == "drain":
        out = drain()
    elif cmd == "fingerprint":
        s, body = call("GET", f"/v1/memories?user_id={USER}&limit=500")
        rows = body if isinstance(body, list) else []
        ids = sorted(r.get("id", "") for r in rows)
        content = sorted(r.get("memory", "") for r in rows)
        out = {"status": s, "count": len(rows),
               "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
               "content_sha256": hashlib.sha256("\n".join(content).encode()).hexdigest()[:16]}
    elif cmd == "search":
        s, body = call("POST", "/v1/search", {"query": sys.argv[2], "user_id": USER, "limit": 20})
        hits = body.get("results", body.get("memories", [])) if isinstance(body, dict) else []
        out = {"status": s, "hits": len(hits)}
    print(json.dumps(out))


main()
