"""Install, upgrade, rollback and uninstall journeys across two images (host side).

Usage: ``journey.py SCENARIO OLD_IMAGE NEW_IMAGE OUT.json`` with SCENARIO one of
``default`` (default flags, data preserved across image changes), ``ledger-sqlite`` and
``ledger-pg`` (durable ledger enabled on the new image, then rolled back to the old one).
Every step records its seconds, outcome and error log lines.  The compose project has its
own volumes; unrelated volumes are checked before and after.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
PROJECT = "ns-g-p6j"
FILES = ["-f", str(HERE / "compose.yml"), "-f", str(HERE / "journey.override.yml")]
APP = ["api", "worker", "graph-worker", "ingest-worker"]
PG_PASSWORD = "bench-owner-password"
PG_DSN = "host=pg dbname=postgres user=ns_owner password=" + PG_PASSWORD + " options='-c search_path=ledger'"


def run(*args, env=None, check=True):
    done = subprocess.run(args, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and done.returncode:
        raise RuntimeError(f"{' '.join(args[:5])}: {done.stderr[-300:]}")
    return done


class Journey:
    def __init__(self, old, new, out):
        self.images = {"old": old, "new": new}
        self.out = out
        self.steps = []
        self.flags = {}
        self.image = old

    def env(self):
        return {"NS_IMAGE": self.image, "BENCH_DIR": str(BENCH), "OUT_DIR": "/tmp", "TOKEN_SECRET": "",
                **self.flags}

    def compose(self, *args, check=True):
        return run("docker", "compose", "-p", PROJECT, *FILES, "--profile", "ledger", "--profile", "pg", *args,
                   env=self.env(), check=check)

    def client(self, *args):
        done = self.compose("run", "--rm", "--no-deps", "--entrypoint", "python", "-v",
                            f"{HERE / 'journey_client.py'}:/c.py:ro", "driver", "/c.py", *args)
        return json.loads(done.stdout.strip().splitlines()[-1])

    def wait_live(self, timeout=300):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            p = self.compose("exec", "-T", "api", "python", "-c",
                             "import urllib.request;urllib.request.urlopen('http://localhost:8199/health/live',timeout=3)", check=False)
            if p.returncode == 0:
                return round(time.monotonic() - start, 1)
            time.sleep(1)
        raise RuntimeError("api never live")

    def error_lines(self, since):
        lines = []
        for svc in APP + (["ledger-worker"] if self.flags.get("LEDGER_ENABLED") == "true" else []):
            out = self.compose("logs", "--no-color", "--since", since, svc, check=False).stdout
            for line in out.splitlines():
                low = line.lower()
                if "traceback" in line or " error" in low or "critical" in low or "exception" in low:
                    lines.append(f"{svc}: {line.strip()[:200]}")
        return lines[:8], len(lines)

    def step(self, name, fn):
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        start = time.monotonic()
        entry = {"step": name, "image": self.image}
        try:
            entry["result"] = fn()
            entry["ok"] = True
        except Exception as e:  # noqa: BLE001 - recorded
            entry["ok"] = False
            entry["error"] = str(e)[:300]
        entry["seconds"] = round(time.monotonic() - start, 1)
        try:
            sample, count = self.error_lines(since)
            entry["log_error_lines"] = count
            entry["log_error_sample"] = sample
        except Exception as e:  # noqa: BLE001
            entry["log_error_lines"] = f"unavailable: {e}"
        self.steps.append(entry)
        Path(self.out).write_text(json.dumps({"steps": self.steps, "flags": {k: v for k, v in self.flags.items() if "PASSWORD" not in k}}, indent=2))
        print(f"[{entry['seconds']:>6}s] {name}: {'ok' if entry['ok'] else 'FAILED ' + entry.get('error', '')}", flush=True)
        return entry

    # -- actions ----------------------------------------------------------
    def start_data(self):
        pg = ["pg"] if self.flags.get("LEDGER_BACKEND") == "postgres" else []
        self.compose("up", "-d", "--wait", "neo4j", "redis", "qdrant", "fake-provider", *pg)
        if pg:
            self.compose("exec", "-T", "pg", "psql", "-U", "postgres", "-c",
                         f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='ns_owner') THEN "
                         f"CREATE ROLE ns_owner LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{PG_PASSWORD}'; END IF; END $$; "
                         "CREATE SCHEMA IF NOT EXISTS ledger AUTHORIZATION ns_owner;")

    def start_app(self):
        names = APP + (["ledger-worker"] if self.flags.get("LEDGER_ENABLED") == "true" else [])
        self.compose("up", "-d", "--force-recreate", *names)
        return {"live_seconds": self.wait_live()}

    def stop_app(self):
        self.compose("stop", *APP, "ledger-worker", check=False)

    def volumes(self):
        out = run("docker", "volume", "ls", "--format", "{{.Name}}").stdout.split()
        return sorted(out)

    def sqlite_versions(self):
        p = self.compose("exec", "-T", "api", "python", "-c",
                         "import sqlite3,json;c=sqlite3.connect('/state/ledger.db');print(json.dumps(c.execute('select version,name,applied_at from ledger_schema_version order by version').fetchall()))",
                         check=False)
        return json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 and p.stdout.strip() else {"error": p.stderr[-200:]}

    def pg_versions(self):
        p = self.compose("exec", "-T", "pg", "psql", "-U", "postgres", "-tA", "-c",
                         "select version||'|'||name||'|'||applied_at from ledger.ledger_schema_version order by version", check=False)
        return p.stdout.split() if p.returncode == 0 else {"error": p.stderr[-200:]}

    def versions(self):
        return self.pg_versions() if self.flags.get("LEDGER_BACKEND") == "postgres" else self.sqlite_versions()

    def intent_counts(self):
        if self.flags.get("LEDGER_BACKEND") == "postgres":
            p = self.compose("exec", "-T", "pg", "psql", "-U", "postgres", "-tA", "-c",
                             "select status||':'||count(*) from ledger.intents group by status", check=False)
            return p.stdout.split()
        p = self.compose("exec", "-T", "api", "python", "-c",
                         "import sqlite3,json;c=sqlite3.connect('/state/ledger.db');print(json.dumps(c.execute('select status,count(*) from intents group by status').fetchall()))",
                         check=False)
        return json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"error": p.stderr[-200:]}


def pending_scenario(old, new, out):
    """Accepted-but-unprocessed intents must survive an image rollback and be applied after re-upgrade."""
    j = Journey(old, new, out)
    j.flags = {"LEDGER_ENABLED": "true"}
    j.image = new
    j.step("start datastores", lambda: j.start_data())
    j.step("prepare shared state volume", lambda: j.compose("run", "--rm", "--no-deps", "--user", "0", "--entrypoint", "sh", "seed",
                                                          "-c", "mkdir -p /state && chmod 777 /state") and "ok")
    j.step("new: start application", j.start_app)
    j.step("new: stop the ledger worker", lambda: j.compose("stop", "ledger-worker") and "stopped")
    j.step("new: accept 10 writes with no worker", lambda: j.client("writenodrain", "10", "pending"))
    j.step("new: ledger intents (expect pending)", j.intent_counts)
    j.step("new: fingerprint (expect nothing stored yet)", lambda: j.client("fingerprint"))
    j.stop_app()
    j.image = old
    j.step("rollback: start application (old image)", j.start_app)
    j.step("old: fingerprint (pending writes are invisible, not lost)", lambda: j.client("fingerprint"))
    j.step("old: ledger intents", j.intent_counts)
    j.stop_app()
    j.image = new
    j.step("re-upgrade: start application (new image)", j.start_app)
    j.step("re-upgrade: drain", lambda: j.client("drain"))
    j.step("new: ledger intents (expect applied)", j.intent_counts)
    j.step("new: fingerprint (expect 10 memories)", lambda: j.client("fingerprint"))
    j.step("uninstall: compose down -v", lambda: j.compose("down", "-v", "--remove-orphans") and "ok")
    Path(out).write_text(json.dumps({"steps": j.steps, "flags": {k: v for k, v in j.flags.items() if "PASSWORD" not in k}}, indent=2))


def main():
    scenario, old, new, out = sys.argv[1:5]
    if scenario == "ledger-pending":
        return pending_scenario(old, new, out)
    j = Journey(old, new, out)
    sentinel = "ns-g-p6j-sentinel"
    run("docker", "volume", "create", sentinel)
    before = j.volumes()
    # state: the shared volume must be writable by the service user
    ledger = scenario.startswith("ledger")
    if ledger:
        j.flags = {"LEDGER_ENABLED": "true"}
        if scenario == "ledger-pg":
            j.flags.update({"LEDGER_BACKEND": "postgres", "LEDGER_POSTGRES_DSN": PG_DSN})
    if scenario == "default":
        j.image = old
    else:
        j.image = old  # install and seed on the old image first (ledger flags are ignored by it)
    j.step("start datastores", lambda: j.start_data())
    j.step("prepare shared state volume", lambda: j.compose("run", "--rm", "--no-deps", "--user", "0", "--entrypoint", "sh", "seed",
                                                          "-c", "mkdir -p /state && chmod 777 /state") and "ok")
    j.step("install: start application (old image, empty volumes)", j.start_app)
    j.step("old: write 30 + drain", lambda: j.client("write", "30", "old"))
    f0 = {}
    j.step("old: fingerprint", lambda: f0.update(j.client("fingerprint")) or dict(f0))
    j.step("old: search", lambda: j.client("search", "Journey fact old"))
    j.stop_app()
    j.image = new
    j.step("upgrade: start application (new image, same volumes)", j.start_app)
    f1 = {}
    j.step("new: fingerprint after upgrade", lambda: f1.update(j.client("fingerprint")) or {**f1, "same_as_old": f1.get("ids_sha256") == f0.get("ids_sha256") and f1.get("content_sha256") == f0.get("content_sha256")})
    j.step("new: search", lambda: j.client("search", "Journey fact old"))
    if ledger:
        j.step("new: ledger schema versions (after first start)", j.versions)
    j.step("new: write 20 + drain", lambda: j.client("write", "20", "new"))
    f2 = {}
    j.step("new: fingerprint", lambda: f2.update(j.client("fingerprint")) or dict(f2))
    if ledger:
        j.step("new: ledger intents", j.intent_counts)
    j.step("restart new image twice", lambda: [j.start_app(), j.start_app()])
    if ledger:
        j.step("new: ledger schema versions (after two restarts)", j.versions)
    f3 = {}
    j.step("new: fingerprint after restarts", lambda: f3.update(j.client("fingerprint")) or {**f3, "same_as_before_restarts": f3 == f2})
    j.stop_app()
    j.image = old
    j.step("rollback: start application (old image, same volumes)", j.start_app)
    f4 = {}
    j.step("old (rolled back): fingerprint", lambda: f4.update(j.client("fingerprint")) or {**f4, "same_as_new": f4.get("ids_sha256") == f2.get("ids_sha256")})
    j.step("old (rolled back): write 5 + drain", lambda: j.client("write", "5", "back"))
    if ledger:
        j.step("old: ledger intents after rollback", j.intent_counts)
    j.stop_app()
    j.image = new
    j.step("re-upgrade: start application (new image)", j.start_app)
    f5 = {}
    j.step("new (re-upgraded): fingerprint", lambda: f5.update(j.client("fingerprint")) or dict(f5))
    mid = j.volumes()
    ids = run("docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={PROJECT}").stdout.split()
    mounted = set()
    if ids:
        info = json.loads(run("docker", "inspect", *ids).stdout)
        mounted = {m["Name"] for c in info for m in c.get("Mounts", []) if m.get("Type") == "volume"}
    j.step("uninstall: compose down -v", lambda: j.compose("down", "-v", "--remove-orphans") and "ok")
    after = j.volumes()
    own_before = [v for v in mid if v.startswith(PROJECT + "_")]
    j.steps.append({"step": "uninstall check", "project_volumes_before_down": own_before,
                    "project_volumes_after_down": [v for v in after if v.startswith(PROJECT + "_")],
                    "other_volumes_unchanged": [v for v in before if v in after] == before or sorted(set(before) - set(after)) == [],
                    "sentinel_present": sentinel in after,
                    "volumes_removed_not_project_named": sorted(v for v in set(mid) - set(after) if not v.startswith(PROJECT + "_")),
                    "removed_volumes_mounted_by_project_containers": sorted(v for v in set(mid) - set(after) if v in mounted),
                    "removed_volumes_not_project_named_and_not_mounted_by_project": sorted(v for v in set(mid) - set(after) if not v.startswith(PROJECT + "_") and v not in mounted)})
    run("docker", "volume", "rm", sentinel, check=False)
    Path(out).write_text(json.dumps({"steps": j.steps, "flags": {k: v for k, v in j.flags.items() if "PASSWORD" not in k}}, indent=2))


main()
