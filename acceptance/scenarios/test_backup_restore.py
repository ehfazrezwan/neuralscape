"""Backup, then restore, on the stack's own volumes (atlas ``ops.backups``).

DESTRUCTIVE. Runs only with ``NS_ALLOW_DESTRUCTIVE=1``, ``--compose-project`` naming
a disposable project (``guard.check_compose_project``), and the docker CLI. It wipes
every memory on the stack (the Qdrant collections, Redis, the ingest artifacts), so
it runs last (``RUN_LAST``). The backup lands in a fresh ``nsc-backup-*`` directory
under the temp dir; anywhere else is refused.

The story, one module fixture (``story``) that records what happened and never
asserts; the scenarios assert:

 1. user A writes through the real surfaces: a raw memory (vector store), waits for it
    in the graph listing (Neo4j), uploads a small file (artifact on the files volume),
    and keeps the memory's task id (task status lives in Redis);
 2. ``scripts/backup-volumes.sh -p <project> --quiesce -d <tmp>`` (+ the gate's
    ``NS_COMPOSE_FILES`` as ``-f``): exit 0, a complete ``ns-backup-<stamp>``, a
    MANIFEST, ``sha256sum -c SHA256SUMS`` passes;
 3. a second memory is written AFTER the backup (the negative control);
 4. wipe: bulk-delete the project through the API (vector + graph), stop the app, delete
    the Qdrant collections, FLUSHALL Redis, wipe /data/ingest, start the app: the markers,
    the task status and the artifact are all GONE (otherwise the restore proves nothing);
 5. restore per the script's RESTORE section: stop the app, ``neo4j-admin database load``,
    Qdrant snapshot upload, ``dump.rdb`` back into Redis, the files tar, start everything;
 6. the first marker is back in vector recall, the graph listing, the task status and the
    artifact download; the post-backup marker is NOT; nothing restarted by itself; a
    fresh write works.

Whatever happens, a finalizer starts every service again and fails with ``compose ps``
if the stack does not come back healthy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept import restore, stack
from nsaccept.flow import Flow, run_flow
from nsaccept.client import NSClient
from nsaccept.helpers import blob_has, graph_listing, memory_ids_in
from nsaccept.runctx import app_services
from nsaccept.tokens import scrub
from nsaccept.upload import upload_files

GRAPH_WAIT = 300
SEARCH_WAIT = 120
TASK_WAIT = 180
BACKUP_TIMEOUT = 1800
STEP_TIMEOUT = 900
QDRANT_COLLECTION = os.environ.get("NS_QDRANT_COLLECTION", "neuralscape_memories")
PARTS = {"redis", "qdrant", "files", "neo4j"}


# ── the real checks (positive scenarios and negative controls share them) ──


def check_vector_has(client: NSClient, marker: str, project_id: str) -> None:
    results = client.search(f"{marker} harbour lamp oil", project_id=project_id, limit=20)
    assert any(marker in r.get("memory", "") for r in results), f"{marker} not in {len(results)} search results"


def check_graph_has(client: NSClient, marker: str, project_id: str) -> None:
    episodes = graph_listing(client, "episodes", project_id)
    assert blob_has(episodes, marker), f"no graph episode carries {marker} among {len(episodes)} episodes"


def check_task_completed(client: NSClient, task_id: str) -> None:
    status = client.task_status(task_id)
    assert status.get("status") == "completed", f"task {task_id} is {status.get('status')!r}"


def check_artifact_bytes(client: NSClient, file_id: str, expected: bytes) -> None:
    r = client.get(f"/v1/ingest/artifacts/{file_id}")
    assert r.status_code == 200, f"GET artifact -> {r.status_code}: {r.text[:120]}"
    assert r.content == expected, "downloaded bytes differ from the uploaded file"


def check_gone(check, *args) -> None:
    """Passes only when the real ``check`` FAILS (the thing is gone)."""
    try:
        check(*args)
    except AssertionError:
        return
    raise AssertionError(f"{check.__name__} still passes: the data is still there")


def check_backup_layout(root: Path, db: str = "memory", collection: str | None = None) -> Path:
    """One complete ``ns-backup-<stamp>`` (no ``.partial`` left), MANIFEST with every
    part, every part's files present and non-empty. Returns the backup directory."""
    root = Path(root)
    partial = [p.name for p in root.iterdir() if p.name.endswith(".partial")]
    assert not partial, f"partial backup left behind: {partial}"
    done = [p for p in root.iterdir() if p.is_dir() and restore.BACKUP_DIR_RE.fullmatch(p.name)]
    assert len(done) == 1, f"expected exactly one complete ns-backup-<stamp>, found {[p.name for p in done]}"
    b = done[0]
    manifest = b / "MANIFEST.txt"
    assert manifest.is_file(), "MANIFEST.txt missing"
    parts = set()
    for line in manifest.read_text().splitlines():
        if line.startswith("parts="):
            parts = set(line[len("parts="):].split())
    assert PARTS <= parts, f"MANIFEST parts={sorted(parts)} lacks {sorted(PARTS - parts)}"
    assert (b / "SHA256SUMS").is_file(), "SHA256SUMS missing"
    for rel in ("redis/dump.rdb", f"neo4j/{db}.dump", "qdrant/index.json", f"files/{restore.FILES_TAR}"):
        f = b / rel
        assert f.is_file() and f.stat().st_size > 0, f"{rel} missing or empty"
    index = json.loads((b / "qdrant" / "index.json").read_text())
    assert index, "qdrant/index.json lists no collections"
    if collection is not None:
        assert collection in index, f"qdrant/index.json lacks the memories collection {collection!r}: {sorted(index)}"
    for name, fname in index.items():
        f = b / "qdrant" / fname
        assert f.is_file() and f.stat().st_size > 0, f"qdrant snapshot for {name} missing or empty"
    return b


def check_checksums(b: Path) -> None:
    """``sha256sum -c SHA256SUMS`` (``shasum -a 256 -c`` where there is no sha256sum)."""
    cmd = ["sha256sum", "-c", "SHA256SUMS"] if shutil.which("sha256sum") else ["shasum", "-a", "256", "-c", "SHA256SUMS"]
    proc = subprocess.run(cmd, cwd=str(b), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"checksum verification failed: {(proc.stdout + proc.stderr)[-300:]}"


# ── the story ────────────────────────────────────────────────────


@dataclass
class Story:
    project_id: str
    markers: dict = field(default_factory=dict)  # m1 (before the backup), m2 (after), m3 (fresh write), file
    task_ids: dict = field(default_factory=dict)
    file_id: str = ""
    file_bytes: bytes = b""
    root: Path | None = None
    backup_dir: Path | None = None
    steps: dict = field(default_factory=dict)  # the Flow's recorded results (name -> "" or failure text)
    backup_exit: int | None = None
    backup_tail: str = ""
    restarts_before: dict = field(default_factory=dict)
    restarts_after: dict = field(default_factory=dict)
    ran_wipe: bool = False
    ran_restore: bool = False


def _write_memory(a: NSClient, story: Story, key: str, project_id: str) -> None:
    marker = story.markers[key]
    task = a.store_raw(
        f"{marker} The harbour lamp oil is stored in the third cellar.", "tech_stack",
        scope="project", project_id=project_id, visibility="private", tags=["acceptance"],
    )
    story.task_ids[key] = task
    status = a.poll_task(task, timeout=TASK_WAIT)
    assert status.get("status") == "completed", f"write {key} ended as {status}"
    assert memory_ids_in(status), f"write {key} completed without a memory: {status}"


def _seed(ctx, a: NSClient, story: Story) -> None:
    _write_memory(a, story, "m1", story.project_id)
    eventually(lambda: check_vector_has(a, story.markers["m1"], story.project_id), timeout=SEARCH_WAIT, what="m1 in vector search")
    marker = story.markers["file"]
    story.file_bytes = f"# Mooring log\n\n{marker} The lamp oil barrels are rolled out on Fridays.\n".encode()
    r = upload_files(
        a, {f"{marker}.md": story.file_bytes}, mime="text/markdown", category="tech_stack", scope="project",
        project_id=story.project_id, visibility="private", extract_facts=False,
    )
    info = r.json()
    story.file_id = info["files"][0]["file_id"]
    story.task_ids["file"] = info["files"][0]["task_id"]
    status = a.poll_task(story.task_ids["file"], timeout=TASK_WAIT)
    assert status.get("status") == "completed", f"file ingest ended as {status}"
    check_artifact_bytes(a, story.file_id, story.file_bytes)
    eventually(lambda: check_vector_has(a, marker, story.project_id), timeout=SEARCH_WAIT, what="file passage in vector search")
    eventually(lambda: check_graph_has(a, story.markers["m1"], story.project_id), timeout=GRAPH_WAIT, interval=5, what="m1 in the graph listing")


def _backup(ctx, story: Story) -> None:
    proc = restore.run_backup(ctx.compose_project, story.root, timeout=BACKUP_TIMEOUT)
    story.backup_exit = proc.returncode
    story.backup_tail = scrub((proc.stderr or "")[-400:])
    assert proc.returncode == 0, f"backup-volumes.sh exited {proc.returncode}: {story.backup_tail}"
    story.backup_dir = check_backup_layout(story.root, restore.neo4j_database(), QDRANT_COLLECTION)  # before any wipe
    out = (proc.stdout or "").strip().splitlines()
    assert out and Path(out[-1]).resolve() == story.backup_dir.resolve(), f"script printed {out[-1:]!r}, found {story.backup_dir}"
    check_checksums(story.backup_dir)


def _wipe(ctx, a: NSClient, story: Story) -> None:
    root = str(story.root)
    story.ran_wipe = True
    r = a.request("DELETE", "/v1/memories", json={"project_id": story.project_id})
    assert r.status_code == 200, f"bulk DELETE -> {r.status_code}: {r.text[:200]}"
    eventually(
        lambda: check_gone(check_graph_has, a, story.markers["m1"], story.project_id),
        timeout=120, what="graph episodes gone after the bulk delete",
    )
    ctx.compose("stop", *app_services(), timeout=300)
    base = os.environ.get("NS_QDRANT_COLLECTION", "neuralscape_memories")
    for name in (base, f"{base}_entities"):
        stack.qdrant_http(ctx, "DELETE", f"/collections/{name}")
    ctx.compose("exec", "-T", restore.redis_service(), "redis-cli", "FLUSHALL")
    ctx.compose(
        "run", "--rm", "--no-deps", "-T", "--entrypoint", "sh", restore.api_service(), "-c", restore.WIPE_INGEST_SCRIPT,
        backup_root=root, timeout=300,
    )
    ctx.compose("up", "-d", "--no-build", "--no-recreate", "--no-deps", *app_services(), timeout=300)
    stack.wait_ready(ctx, a, "the wipe")


def _prove_gone(a: NSClient, story: Story) -> None:
    """Before restoring: every marker is unreachable through every surface."""
    pid = story.project_id
    # The surfaces must ANSWER (a 5xx would also make a check "fail"): search 200, graph listing 200,
    # the task status endpoint answers 404/not_found, the artifact download says 404.
    a.search("readiness probe", project_id=pid, limit=1)
    graph_listing(a, "episodes", pid)
    # An unknown task is a 404 on the status route ("404 when unknown or not theirs"); a 200 body saying
    # not_found is the same answer. Anything else means the task is still known.
    resp = a.get(f"/v1/memories/status/{story.task_ids['m1']}")
    state = "not_found" if resp.status_code == 404 else (resp.json().get("status") if resp.status_code == 200 else f"HTTP {resp.status_code}")
    assert state == "not_found", f"task status after FLUSHALL is {state!r}, want not_found (404)"
    code = a.get(f"/v1/ingest/artifacts/{story.file_id}").status_code
    assert code == 404, f"artifact download after the wipe -> {code}, want 404"
    for key in ("m1", "m2"):
        check_gone(check_vector_has, a, story.markers[key], pid)
    check_gone(check_vector_has, a, story.markers["file"], pid)
    check_gone(check_graph_has, a, story.markers["m1"], pid)
    check_gone(check_task_completed, a, story.task_ids["m1"])
    check_gone(check_artifact_bytes, a, story.file_id, story.file_bytes)


def _restore(ctx, a: NSClient, story: Story) -> None:
    root, b = str(story.root), story.backup_dir
    story.ran_restore = True
    ctx.compose("stop", *app_services(), timeout=300)
    # Neo4j: stop, load the dump over the database, start
    ctx.compose("stop", restore.neo4j_service(), timeout=300)
    ctx.compose(
        "run", "--rm", "--no-deps", "-T", "-v", f"{b}/neo4j:/backups", restore.neo4j_service(),
        "neo4j-admin", "database", "load", restore.neo4j_database(),
        "--from-path=/backups", "--overwrite-destination=true",
        backup_root=root, timeout=STEP_TIMEOUT,
    )
    ctx.compose("start", restore.neo4j_service(), timeout=300)
    stack.wait_services(ctx, [restore.neo4j_service()], "the neo4j load")
    # Qdrant: upload every collection snapshot (priority=snapshot)
    ctx.compose(
        "run", "--rm", "--no-deps", "-T", "-v", f"{b}/qdrant:/backups:ro", restore.api_service(),
        "python", "-c", restore.QDRANT_UPLOAD_SCRIPT,
        backup_root=root, timeout=STEP_TIMEOUT,
    )
    # Redis: stop, put dump.rdb back, start
    ctx.compose("stop", restore.redis_service(), timeout=300)
    ctx.compose("cp", f"{b}/redis/dump.rdb", f"{restore.redis_service()}:/data/dump.rdb", backup_root=root, timeout=120)
    ctx.compose("start", restore.redis_service(), timeout=300)
    stack.wait_services(ctx, [restore.redis_service()], "the redis dump restore")
    # Files: the ingest + exemplar tar, on stdin as the script's RESTORE section does
    ctx.compose(
        "run", "--rm", "--no-deps", "-T", "-i", "--entrypoint", "tar", restore.api_service(),
        "-C", "/data", "-xzf", "-",
        backup_root=root, stdin_path=str(b / "files" / restore.FILES_TAR), timeout=STEP_TIMEOUT,
    )
    ctx.compose("up", "-d", "--no-build", "--no-recreate", "--no-deps", *app_services(), timeout=300)
    stack.wait_ready(ctx, a, "the restore")


def _post_restore(a: NSClient, story: Story) -> None:
    pid, m = story.project_id, story.markers
    eventually(lambda: check_vector_has(a, m["m1"], pid), timeout=SEARCH_WAIT, what="m1 back in vector search")
    eventually(lambda: check_vector_has(a, m["file"], pid), timeout=SEARCH_WAIT, what="file passage back in vector search")


def _post_graph(a: NSClient, story: Story) -> None:
    eventually(lambda: check_graph_has(a, story.markers["m1"], story.project_id), timeout=180, interval=5, what="m1 back in the graph listing")


def _post_task(a: NSClient, story: Story) -> None:
    check_task_completed(a, story.task_ids["m1"])


def _post_artifact(a: NSClient, story: Story) -> None:
    check_artifact_bytes(a, story.file_id, story.file_bytes)


def _fresh_write(a: NSClient, story: Story) -> None:
    _write_memory(a, story, "m3", story.project_id)
    eventually(lambda: check_vector_has(a, story.markers["m3"], story.project_id), timeout=SEARCH_WAIT, what="fresh write recalled")


def _after_backup_marker(ctx, a: NSClient, story: Story) -> None:
    stack.wait_ready(ctx, a, "the backup script (it restarts the app and neo4j)")
    _write_memory(a, story, "m2", story.project_id)
    eventually(lambda: check_vector_has(a, story.markers["m2"], story.project_id), timeout=SEARCH_WAIT, what="m2 in vector search")
    eventually(lambda: check_graph_has(a, story.markers["m2"], story.project_id), timeout=GRAPH_WAIT, interval=5, what="m2 in the graph listing")


def _gate(ctx) -> None:
    reason = restore.gate_reason(os.environ, ctx.compose_project)  # raises for prod / off-allowlist names
    if reason:
        pytest.skip(reason)
    try:
        restore.compose_file_args()
    except restore.RestoreRefused as exc:
        pytest.fail(f"NS_COMPOSE_FILES: {exc}")


def _preflight(ctx, a: NSClient, story: Story) -> None:
    """Before anything is wiped: prove every command shape the restore needs can run."""
    stack.wait_ready(ctx, a, "before the backup scenario")
    restore.backup_command(ctx.compose_project, story.root)  # script present, destination fresh, project allowed
    root = str(story.root)
    ctx.compose("config", "-q")
    ctx.compose("run", "--rm", "--no-deps", "-T", restore.neo4j_service(), "neo4j-admin", "--version", backup_root=root, timeout=300)
    ctx.compose("run", "--rm", "--no-deps", "-T", "--entrypoint", "tar", restore.api_service(), "--version", backup_root=root, timeout=300)
    ctx.compose("run", "--rm", "--no-deps", "-T", restore.api_service(), "python", "-c", restore.QDRANT_PING_SCRIPT, backup_root=root, timeout=300)
    story.restarts_before = stack.restart_counts(ctx)


@pytest.fixture(scope="module")
def story(ctx, a):
    _gate(ctx)
    s = Story(project_id=f"{ctx.project_id}-bkp")
    s.markers = {k: f"zq-{ctx.run_id}-bk{k}" for k in ("m1", "m2", "m3", "file")}
    s.root = restore.make_backup_root()
    flow = Flow()
    s.steps = flow.steps

    def wipe():
        s.ran_wipe = True
        _wipe(ctx, a, s)

    def do_restore():
        s.ran_restore = True
        _restore(ctx, a, s)

    fns = {
        "preflight": lambda: _preflight(ctx, a, s),
        "seed": lambda: _seed(ctx, a, s),
        "backup": lambda: _backup(ctx, s),
        "after_backup_write": lambda: _after_backup_marker(ctx, a, s),
        "wipe": wipe,
        "gone": lambda: _prove_gone(a, s),
        "restore": do_restore,
        "post_vector": lambda: _post_restore(a, s),
        "post_graph": lambda: _post_graph(a, s),
        "post_task": lambda: _post_task(a, s),
        "post_artifact": lambda: _post_artifact(a, s),
        "restarts": lambda: setattr(s, "restarts_after", stack.restart_counts(ctx)),
        "fresh_write": lambda: _fresh_write(a, s),
    }
    try:
        run_flow(flow, fns)
        if flow.keep_backup:
            # The backup is the only copy of the data: leave it for the operator, say where.
            note = (f"; the backup is KEPT at {s.root} (restore it by hand per scripts/backup-volumes.sh, "
                    "then remove that directory)")
            s.steps["restore"] = (s.steps.get("restore") or "restore did not run") + note
        yield s
    finally:
        try:
            stack.ensure_running(ctx, a, "the backup/restore finalizer")
        finally:
            if not flow.keep_backup:
                shutil.rmtree(s.root, ignore_errors=True)


def require(story: Story, *names: str) -> None:
    assert not story.steps.get("preflight", ""), f"preflight failed, nothing was wiped: {story.steps.get('preflight')}"
    for n in names:
        assert n in story.steps, f"step {n!r} never ran (an earlier step failed: {story.steps})"
        assert not story.steps[n], f"step {n!r} failed: {story.steps[n]}"


# ── scenarios ────────────────────────────────────────────────────


@proves("ops.backups")
@needs("compose")
def test_backup_script_writes_a_complete_verified_backup_of_all_four_parts(story):
    require(story, "seed")
    assert story.backup_exit == 0, f"backup exit {story.backup_exit}: {story.backup_tail}"
    require(story, "backup")
    b = check_backup_layout(story.root, restore.neo4j_database())
    check_checksums(b)


@proves("ops.backups", "queue.task-manager", "ingest.artifact-storage", "read.graph-listing", "read.hybrid-search-core")
@needs("compose")
def test_the_data_was_visible_in_every_store_before_the_backup(story):
    require(story, "seed")  # vector, graph episode, artifact bytes, completed task status all checked in the seed step


@proves("ops.backups", "queue.task-manager", "ingest.artifact-storage", "read.graph-listing", "read.hybrid-search-core")
@needs("compose")
def test_the_wipe_removed_the_marker_from_vector_graph_task_status_and_artifact(story):
    require(story, "backup")
    require(story, "after_backup_write", "wipe", "gone")


@proves("ops.backups", "read.hybrid-search-core", "read.graph-listing", "ingest.artifact-storage")
@needs("compose")
def test_restore_brings_back_vector_recall_graph_listing_and_the_artifact(story):
    require(story, "gone", "restore", "post_vector", "post_graph", "post_artifact")


@proves("ops.backups", "queue.task-manager")
@needs("compose")
def test_restore_brings_back_the_task_status_from_redis(story):
    require(story, "gone", "restore", "post_task")


@proves("ops.backups", "read.hybrid-search-core", "read.graph-listing")
@negative_control
@needs("compose")
def test_a_marker_written_after_the_backup_is_absent_after_the_restore(ctx, a, story):
    """The same checks that pass for the backed-up marker FAIL for the one written afterwards:
    the restore rolled the stores back to the backup, it was not just that nothing had been wiped."""
    require(story, "after_backup_write", "restore")
    expect_check_fails(check_vector_has, a, story.markers["m2"], story.project_id)
    expect_check_fails(check_graph_has, a, story.markers["m2"], story.project_id)
    expect_check_fails(check_task_completed, a, story.task_ids["m2"])


@proves("ops.backups")
@needs("compose")
def test_stack_is_healthy_and_nothing_restarted_by_itself_after_the_restore(ctx, story):
    require(story, "restore", "restarts")
    services = stack.all_services()
    stack.check_no_new_restarts(story.restarts_before, story.restarts_after, services)
    assert stack.rows_ready(stack.ps_rows(ctx, services), services) is None, "a service is not running and healthy"


@proves("ops.backups", "write.remember-raw")
@needs("compose")
def test_a_fresh_write_works_after_the_restore(story):
    require(story, "restore", "fresh_write")


# ── negative controls that need no stack ─────────────────────────


@proves("ops.backups")
@negative_control
def test_layout_check_fails_for_partial_empty_or_incomplete_backups(tmp_path):
    root = tmp_path / "nsc-backup-x"
    root.mkdir()
    expect_check_fails(check_backup_layout, root)  # nothing there
    (root / "ns-backup-20260101T000000Z.partial").mkdir()
    expect_check_fails(check_backup_layout, root)  # only a partial
    (root / "ns-backup-20260101T000000Z.partial").rmdir()
    good = root / "ns-backup-20260101T000000Z"
    for rel, body in (("redis/dump.rdb", "r"), ("neo4j/memory.dump", "n"), ("files/" + restore.FILES_TAR, "f"),
                      ("qdrant/index.json", json.dumps({"c": "c.snap"})), ("qdrant/c.snap", "q"), ("SHA256SUMS", "x")):
        (good / rel).parent.mkdir(parents=True, exist_ok=True)
        (good / rel).write_text(body)
    expect_check_fails(check_backup_layout, root)  # no MANIFEST
    (good / "MANIFEST.txt").write_text("parts=redis qdrant files\n")
    expect_check_fails(check_backup_layout, root)  # neo4j missing from the manifest
    (good / "MANIFEST.txt").write_text("parts=redis qdrant files neo4j\n")
    assert check_backup_layout(root) == good  # complete: passes
    assert check_backup_layout(root, collection="c") == good
    expect_check_fails(check_backup_layout, root, "memory", "neuralscape_memories")  # the memories collection is not in the backup
    (good / "redis" / "dump.rdb").write_text("")
    expect_check_fails(check_backup_layout, root)  # an empty part
    (good / "redis" / "dump.rdb").write_text("r")
    (good / "qdrant" / "c.snap").unlink()
    expect_check_fails(check_backup_layout, root)  # index names a snapshot that is not there


@proves("ops.backups")
@negative_control
def test_checksum_check_fails_for_a_corrupted_file(tmp_path):
    f = tmp_path / "part.bin"
    f.write_bytes(b"original")
    tool = ["sha256sum"] if shutil.which("sha256sum") else ["shasum", "-a", "256"]
    (tmp_path / "SHA256SUMS").write_text(subprocess.run([*tool, "part.bin"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout)
    check_checksums(tmp_path)
    f.write_bytes(b"tampered")
    expect_check_fails(check_checksums, tmp_path)


@proves("ops.backups")
@negative_control
def test_gone_check_fails_while_the_data_is_still_there():
    def present():
        return None

    expect_check_fails(check_gone, present)
    check_gone(lambda: (_ for _ in ()).throw(AssertionError("absent")))


@proves("ops.backups")
@negative_control
def test_restart_check_fails_when_a_service_restarted():
    svcs = ["a", "b"]
    stack.check_no_new_restarts({"a": 0, "b": 1}, {"a": 0, "b": 1}, svcs)
    expect_check_fails(stack.check_no_new_restarts, {"a": 0, "b": 1}, {"a": 1, "b": 1}, svcs)
    expect_check_fails(stack.check_no_new_restarts, {"a": 0}, {"a": 0}, svcs)
