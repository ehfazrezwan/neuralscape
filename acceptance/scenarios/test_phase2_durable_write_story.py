"""The durable write story, end to end, on the Phase 2 stack (profile ``phase2``).

One user saves a fact, sees it recalled, corrects it, narrows who may read it, then someone replays the
old requests, and finally the record is exported and restored. Each step is its own scenario, all reading
what ONE module fixture recorded (the fixture records and never asserts, so a failing step reads ``fail``
and not ``error``).

Surfaces: ``POST /v1/memories/raw`` (with an ``Idempotency-Key``), ``GET /v1/memories/status/{id}``,
``POST /v1/search``, ``GET /v1/memories/{id}``, ``PATCH /v1/memories/{id}``, ``GET /v1/export/okf``,
``POST /v1/ingest/files`` (an OKF bundle), ``POST /v1/portability/exports``.

Some steps are written against behaviour a later slice delivers and carry ``@expected_red(...)`` with the
slice: they run, they fail honestly today, and they are never skipped or loosened.
"""

from __future__ import annotations

import io
import os
import time
import uuid
import zipfile
from dataclasses import dataclass, field

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, profiles, proves
from nsaccept.redmark import expected_red
from nsaccept.upload import upload_files

from phase2_kit import (
    DENIED, RED_DURABLE_WRITES, RED_PORTABILITY, Step, bundle_text, call, check_blind_by_id,
    check_export_carries, check_not_recalled, check_recalled, memory_id_of, okf_export, receipt_of, revision_of,
)

pytestmark = profiles("phase2")


@dataclass
class Story:
    old: str
    new: str
    ctl: str
    key: str
    save: Step = field(default_factory=lambda: Step("save"))
    memory_id: str = ""
    rev1: int | None = None  # the revision the save's receipt names (None until the write path issues receipts)
    ctl_id: str = ""
    a_saw_before: str = ""  # every "" below means the check held at that moment, else why not
    b_saw_before: str = ""
    correct: Step = field(default_factory=lambda: Step("correct"))
    rev2: int | None = None
    row_after_correct: Step = field(default_factory=lambda: Step("row_after_correct"))
    recall_after_correct: str = ""
    narrow: Step = field(default_factory=lambda: Step("narrow"))
    rev3: int | None = None
    row_after_narrow: Step = field(default_factory=lambda: Step("row_after_narrow"))
    a_recalls_after_narrow: str = ""
    b_blind_after_narrow: str = ""
    export_a: bytes = b""
    export_b: bytes = b""
    export_error: str = ""
    replay_write: Step = field(default_factory=lambda: Step("replay_write"))
    replay_edit: Step = field(default_factory=lambda: Step("replay_edit"))
    row_after_replay: Step = field(default_factory=lambda: Step("row_after_replay"))
    old_text_after_replay: str = ""  # "" = nobody recalls the replaced text
    b_after_replay: str = ""  # "" = B still blind
    deleted: str = ""  # "" = the original record is gone
    recalled_after_delete: str = ""  # "" = nobody recalls the text between the delete and the restore
    restore: Step = field(default_factory=lambda: Step("restore"))
    restore_error: str = ""
    a_recalls_restored: str = ""
    b_blind_restored: str = ""
    portable: Step = field(default_factory=lambda: Step("portable_export"))
    portable_download: Step = field(default_factory=lambda: Step("portable_download"))
    restart_error: str = ""  # "" = the API restarted and answered /health again
    status_after_restart: Step = field(default_factory=lambda: Step("status_after_restart"))

    @property
    def q_old(self) -> str:
        return f"{self.old} harbour pilot green lanterns"

    @property
    def q_new(self) -> str:
        return f"{self.new} harbour pilot amber lanterns"

    @property
    def text_old(self) -> str:
        return f"{self.old} The harbour pilot signals with two green lanterns."

    @property
    def text_new(self) -> str:
        return f"{self.new} The harbour pilot signals with two amber lanterns."

    @property
    def save_body(self) -> dict:
        return {
            "content": self.text_old, "category": "domain_knowledge", "scope": "global",
            "visibility": "shared", "tags": ["acceptance"],
        }


def _if_match(rev: int | None) -> dict:
    return {} if rev is None else {"If-Match": f'"{rev}"'}


def _revision_in(step: Step) -> int | None:
    body = step.body if isinstance(step.body, dict) else {}
    return body.get("revision") if isinstance(body.get("revision"), int) else None


def _attempt(check, *args) -> str:
    """'' when the check holds, else why not (the fixture records, the scenario asserts)."""
    try:
        check(*args)
        return ""
    except AssertionError as exc:
        return str(exc)[:200]


def _slice_bundle(data: bytes, marker: str) -> bytes:
    """The root files of an exported OKF bundle plus only the concept documents that carry ``marker``."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            raw = src.read(name)
            if "/" not in name or marker.encode() in raw:
                dst.writestr(name, raw)
    return out.getvalue()


@pytest.fixture(scope="module")
def story(ctx, a, b) -> Story:
    run = ctx.run_id
    nonce = uuid.uuid4().hex[:6]  # a rerun with the same NS_RUN_ID must not see leftover text
    s = Story(old=f"zq-{run}{nonce}-wsold", new=f"zq-{run}{nonce}-wsnew", ctl=f"zq-{run}{nonce}-wsctl", key=f"acc-{run}-{nonce}-story-save")

    # 1. save, with a caller-chosen idempotency key
    s.save = call("save", a, "POST", "/v1/memories/raw", json=s.save_body, headers={"Idempotency-Key": s.key})
    s.memory_id = memory_id_of(s.save.final)
    s.rev1 = revision_of(s.save.final)
    # a second shared fact that is never touched: the control for every "B cannot see it" check
    ctl = call("control", a, "POST", "/v1/memories/raw", json={
        "content": f"{s.ctl} The harbour bell is struck twice at dusk.", "category": "domain_knowledge",
        "scope": "global", "visibility": "shared",
    })
    s.ctl_id = memory_id_of(ctl.final)
    if not s.memory_id:
        return s

    # 2. recall: both users see the shared fact while it is shared
    s.a_saw_before = _attempt(lambda: eventually(lambda: check_recalled(a, s.old, s.q_old), timeout=90, what="A recalls the fact"))
    s.b_saw_before = _attempt(lambda: eventually(lambda: check_recalled(b, s.old, s.q_old), timeout=90, what="B recalls the shared fact"))

    # 3. correction (at the revision the receipt named, when it named one)
    s.correct = call("correct", a, "PATCH", f"/v1/memories/{s.memory_id}", json={"content": s.text_new}, headers=_if_match(s.rev1))
    s.rev2 = _revision_in(s.correct)
    s.recall_after_correct = _attempt(lambda: eventually(
        lambda: (check_recalled(a, s.new, s.q_new), check_not_recalled(a, s.old, s.q_old)), timeout=90, what="A recalls the correction only"))
    s.row_after_correct = call("row_after_correct", a, "GET", f"/v1/memories/{s.memory_id}", poll=False)

    # 4. privacy narrowing: shared -> private
    s.narrow = call("narrow", a, "PATCH", f"/v1/memories/{s.memory_id}", json={"visibility": "private"}, headers=_if_match(s.rev2 or s.rev1))
    s.rev3 = _revision_in(s.narrow)
    s.b_blind_after_narrow = _attempt(lambda: eventually(
        lambda: (check_not_recalled(b, s.new, s.q_new), check_blind_by_id(b, s.memory_id)), timeout=90, what="B is blind after narrowing"))
    s.a_recalls_after_narrow = _attempt(lambda: eventually(lambda: check_recalled(a, s.new, s.q_new), timeout=60, what="A still recalls it"))
    s.row_after_narrow = call("row_after_narrow", a, "GET", f"/v1/memories/{s.memory_id}", poll=False)

    # 5. export (A's own pool, and B's, which must not carry it)
    try:
        s.export_a = okf_export(a)
        s.export_b = okf_export(b)
    except AssertionError as exc:
        s.export_error = str(exc)

    # 6. stale replay: the original save again with its key, then the pre-narrowing state again at the OLD revision
    s.replay_write = call("replay_write", a, "POST", "/v1/memories/raw", json=s.save_body, headers={"Idempotency-Key": s.key})
    s.replay_edit = call("replay_edit", a, "PATCH", f"/v1/memories/{s.memory_id}", json={"visibility": "shared"}, headers=_if_match(s.rev1))
    time.sleep(5)  # a replay that does take effect shows up in the vector store within seconds
    s.row_after_replay = call("row_after_replay", a, "GET", f"/v1/memories/{s.memory_id}", poll=False)
    s.old_text_after_replay = _attempt(lambda: (check_not_recalled(a, s.old, s.q_old), check_not_recalled(b, s.old, s.q_old)))
    s.b_after_replay = _attempt(lambda: (check_not_recalled(b, s.new, s.q_new), check_blind_by_id(b, s.memory_id)))

    # 7. restore: delete the record, then bring it back from the exported bundle
    if s.export_a:
        gone = a.delete(f"/v1/memories/{s.memory_id}")
        s.deleted = "" if gone.status_code in (200, 202) else f"DELETE -> {gone.status_code}: {gone.text[:120]}"
        if not s.deleted:
            s.deleted = _attempt(lambda: eventually(lambda: check_blind_by_id(a, s.memory_id), timeout=60, what="record gone"))
        s.recalled_after_delete = _attempt(check_not_recalled, a, s.new, s.q_new)
        try:
            up = upload_files(a, {f"{s.key}.zip": _slice_bundle(s.export_a, s.new)}, mime="application/zip",
                              category="domain_knowledge", scope="global", visibility="private", extract_facts=False)
            task = up.json()["files"][0]["task_id"]
            s.restore = Step("restore", up.status_code, up.json(), task, a.poll_task(task, timeout=240))
            s.a_recalls_restored = _attempt(lambda: eventually(lambda: check_recalled(a, s.new, s.q_new), timeout=90, what="A recalls the restored record"))
            s.b_blind_restored = _attempt(check_not_recalled, b, s.new, s.q_new)
        except (AssertionError, KeyError, IndexError) as exc:
            s.restore_error = str(exc)[:200]

    # 8. the verified portable export of the tenant (a later slice; recorded so the scenario can assert)
    s.portable = call("portable_export", a, "POST", "/v1/portability/exports", poll=False)
    body = s.portable.body if isinstance(s.portable.body, dict) else {}
    if s.portable.status == 201 and body.get("export_id"):
        try:
            r = a.get(f"/v1/portability/exports/{body['export_id']}/download",
                      headers={"X-Export-Capability": str(body.get("download_capability", ""))})
            s.portable_download = Step("portable_download", r.status_code, None, "", {"bytes": len(r.content)})
        except Exception as exc:  # noqa: BLE001
            s.portable_download = Step("portable_download", error=f"{type(exc).__name__}: {exc}"[:200])

    # 9. the accepted work survives a restart of the API (last step: later modules wait for /health themselves)
    s.restart_error = _restart_api(ctx, a)
    if not s.restart_error:
        s.status_after_restart = call("status_after_restart", a, "GET", f"/v1/memories/status/{s.save.task_id}", poll=False)
    return s


def _restart_api(ctx, client) -> str:
    """Restart the API service of the compose project and wait for it; '' when it is back, else why not."""
    if not ctx.compose_project:
        return "no compose project: a restart needs --compose-project"
    try:
        ctx.compose("restart", os.environ.get("NS_API_SERVICE", "neuralscape"), timeout=240)
    except Exception as exc:  # noqa: BLE001
        return f"restart failed: {type(exc).__name__}: {exc}"[:200]
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            if client.get("/health").status_code == 200:
                return ""
        except Exception:  # noqa: BLE001 - still coming up
            pass
        time.sleep(3)
    return "the API did not answer /health within 180s of the restart"


# ── step 1: save ─────────────────────────────────────────────────


@proves("write.remember-raw", "model.visibility-tiers")
def test_the_save_is_accepted_and_applied(story):
    assert story.save.status == 202, story.save.describe()
    assert story.save.final.get("status") == "completed", f"write task ended as {story.save.final}"
    assert story.memory_id, f"the completed write named no memory: {story.save.final}"


# ── step 2: accepted receipt (ledger) ────────────────────────────


def check_durable_receipt(client, task_id: str, key: str) -> dict:
    status = client.task_status(task_id)
    assert task_id.startswith("intent-"), f"task id {task_id!r} is a legacy queue id, not a durable intent"
    receipt = receipt_of(status)
    assert receipt, f"status carries no receipt: {str(status)[:200]}"
    assert receipt.get("intent_id") == task_id, f"receipt names intent {receipt.get('intent_id')!r}"
    assert receipt.get("idempotency_key") == key, f"receipt names key {receipt.get('idempotency_key')!r}"
    assert receipt.get("status") == "applied", f"receipt status {receipt.get('status')!r}"
    stages = receipt.get("stages") or []
    assert stages and all(st.get("state") == "applied" for st in stages), f"stages: {stages}"
    outs = [o for o in receipt.get("outputs", []) if o.get("kind") == "memory"]
    assert outs and isinstance(outs[0].get("revision"), int) and outs[0]["revision"] >= 1, f"outputs: {outs}"
    return receipt


@proves("archive.phase2-led-a", "archive.phase2-led-b", "contracts.lifecycle")
@expected_red(RED_DURABLE_WRITES)
def test_an_accepted_write_has_a_durable_receipt_naming_its_key_and_revision(a, story):
    assert story.save.ok, story.save.describe()
    check_durable_receipt(a, story.save.task_id, story.key)


@proves("archive.phase2-led-a", "contracts.lifecycle")
@negative_control
@expected_red(RED_DURABLE_WRITES)
def test_receipt_check_fails_for_a_user_who_does_not_own_the_task(a, b, story):
    assert story.save.task_id, story.save.describe()
    check_durable_receipt(a, story.save.task_id, story.key)  # the same check passes for the owner first
    expect_check_fails(check_durable_receipt, b, story.save.task_id, story.key)


# ── step 3: current recall ───────────────────────────────────────


@proves("read.hybrid-search-core", "model.visibility-tiers")
def test_the_saved_fact_is_recalled_by_its_owner_and_by_the_other_user_while_shared(story):
    assert story.memory_id, story.save.describe()
    assert not story.a_saw_before, f"A did not recall its own fact: {story.a_saw_before}"
    assert not story.b_saw_before, f"B did not recall the shared fact: {story.b_saw_before}"


@proves("read.hybrid-search-core", "model.visibility-tiers")
@negative_control
def test_recall_check_fails_for_a_marker_that_was_never_written(a, b, story):
    check_recalled(a, story.ctl, f"{story.ctl} harbour bell dusk")  # the same check passes for a real marker first
    expect_check_fails(check_recalled, b, f"{story.old}-never", story.q_old)


# ── step 4: correction ───────────────────────────────────────────


def check_row_text(row: Step, want: str) -> None:
    assert row.status == 200 and isinstance(row.body, dict), row.describe()
    assert want in row.body.get("memory", ""), f"the row does not carry {want}"


@proves("edit.patch-memory")
def test_the_correction_replaces_the_recalled_text_and_keeps_the_record(story):
    assert story.correct.ok, story.correct.describe()
    if story.correct.task_id:  # accepted for later application: the task must have finished
        assert story.correct.final.get("status") == "completed", f"correction task ended as {story.correct.final}"
    assert not story.recall_after_correct, f"recall after the correction: {story.recall_after_correct}"
    check_row_text(story.row_after_correct, story.new)
    assert story.old not in story.row_after_correct.body["memory"], "the by-id read still shows the old text"
    assert story.row_after_correct.body["id"] == story.memory_id, "the correction made a different record"


@proves("edit.patch-memory")
@negative_control
def test_correction_check_fails_for_the_replaced_text(story):
    check_row_text(story.row_after_correct, story.new)  # the same check passes for the corrected text first
    expect_check_fails(check_row_text, story.row_after_correct, story.old)


@proves("archive.phase2-led-a", "archive.phase2-led-b", "contracts.lifecycle")
@expected_red(RED_DURABLE_WRITES)
def test_the_correction_names_the_revision_it_corrects_and_returns_the_next_one(story):
    assert story.rev1 is not None, "the save's receipt named no revision to correct"
    assert story.correct.status == 202, story.correct.describe()
    assert isinstance(story.correct.body, dict) and story.correct.body.get("status") == "accepted", story.correct.describe()
    assert story.rev2 == story.rev1 + 1, f"correction at revision {story.rev1} returned revision {story.rev2}"
    assert story.correct.final.get("status") == "completed", f"correction task ended as {story.correct.final}"


# ── step 5: privacy narrowing ────────────────────────────────────


@proves("model.visibility-tiers", "edit.patch-memory")
def test_narrowing_to_private_hides_the_record_from_the_other_user_but_not_its_owner(story):
    assert story.narrow.ok, story.narrow.describe()
    assert not story.b_blind_after_narrow, f"B after narrowing: {story.b_blind_after_narrow}"
    assert not story.a_recalls_after_narrow, f"A after narrowing: {story.a_recalls_after_narrow}"
    check_row_text(story.row_after_narrow, story.new)
    assert story.row_after_narrow.body.get("visibility") == "private", f"visibility {story.row_after_narrow.body.get('visibility')!r}"


@proves("model.visibility-tiers")
@negative_control
def test_blindness_check_fails_for_the_record_that_stays_shared(b, story):
    assert story.ctl_id, "the control fact was not written"

    def check_ctl_is_blind(client):
        check_not_recalled(client, story.ctl, f"{story.ctl} harbour bell dusk")
        check_blind_by_id(client, story.ctl_id)

    expect_check_fails(check_ctl_is_blind, b)


# ── step 6: stale replay is denied ───────────────────────────────


@proves("archive.phase2-led-a", "archive.phase2-led-b", "contracts.lifecycle", "model.visibility-tiers")
@expected_red(RED_DURABLE_WRITES)
def test_replaying_the_original_write_with_its_key_resurrects_nothing(story):
    r = story.replay_write
    assert r.status in (200, 202), r.describe()
    assert r.task_id == story.save.task_id and r.task_id.startswith("intent-"), (
        f"the replay was accepted as {r.task_id!r}, not as the original intent {story.save.task_id!r}"
    )
    assert not story.old_text_after_replay, f"the replaced text came back: {story.old_text_after_replay}"
    check_row_text(story.row_after_replay, story.new)
    assert story.row_after_replay.body.get("visibility") == "private", f"visibility after the replay {story.row_after_replay.body.get('visibility')!r}"


@proves("archive.phase2-led-a", "archive.phase2-led-b", "contracts.lifecycle", "model.visibility-tiers")
@expected_red(RED_DURABLE_WRITES)
def test_a_stale_edit_at_the_old_revision_is_refused_and_does_not_widen_visibility(story):
    assert story.rev1 is not None, "no revision to replay an edit against: the save carried no receipt"
    r = story.replay_edit
    assert r.status == 409, r.describe()
    err = ((r.body or {}).get("detail") or {}).get("error") or {}
    assert err.get("code") == "stale_revision", f"refusal: {r.body}"
    assert err.get("current_revision") == (story.rev3 or story.rev2), f"current revision named: {err.get('current_revision')}"
    assert not story.b_after_replay, f"B after the stale edit: {story.b_after_replay}"
    assert story.row_after_replay.body.get("visibility") == "private", story.row_after_replay.describe()


@proves("archive.phase2-led-b", "contracts.lifecycle")
@negative_control
@expected_red(RED_DURABLE_WRITES)
def test_stale_replay_check_fails_for_a_write_with_a_fresh_key(a, story):
    """A write with a NEW key is a new write, not a replay: the replay check must be able to tell the two apart."""

    def check_is_the_original_intent(client):
        r = client.post("/v1/memories/raw", json={**story.save_body, "content": f"{story.ctl}-fresh The pier light is red."},
                        headers={"Idempotency-Key": f"{story.key}-fresh"})
        assert r.status_code == 202 and r.json().get("task_id") == story.save.task_id, "not the original intent"

    def check_replay_key_returns_original(client):
        r = client.post("/v1/memories/raw", json=story.save_body, headers={"Idempotency-Key": story.key})
        assert r.status_code == 202 and r.json().get("task_id") == story.save.task_id, "not the original intent"

    check_replay_key_returns_original(a)  # the same key passes first
    expect_check_fails(check_is_the_original_intent, a)


# ── step 7: export ───────────────────────────────────────────────


@proves("ingest.okf-export", "read.okf-export", "model.visibility-tiers")
def test_the_export_carries_the_corrected_record_and_not_the_replaced_text(story):
    assert not story.export_error, story.export_error
    text = bundle_text(story.export_a)
    assert story.new in text, "the corrected text is not in the owner's export"
    assert story.old not in text, "the replaced text is still in the owner's export"


@proves("ingest.okf-export", "model.visibility-tiers")
def test_the_other_users_export_carries_nothing_of_the_narrowed_record(story):
    assert not story.export_error, story.export_error
    text = bundle_text(story.export_b)
    assert story.new not in text and story.old not in text, "B's export carries the narrowed record"


@proves("ingest.okf-export")
@negative_control
def test_export_check_fails_for_the_other_user(a, b, story):
    check_export_carries(a, story.new)  # the same check passes for the owner first
    expect_check_fails(check_export_carries, b, story.new)


# ── step 8: restore ──────────────────────────────────────────────


@proves("ingest.okf-bundle-import", "ingest.okf-export", "model.visibility-tiers")
def test_the_record_restored_from_the_bundle_is_the_corrected_one_and_stays_private(story):
    assert story.export_a, f"nothing was exported, so nothing was restored: {story.export_error}"
    assert not story.deleted, f"the original record was not removed before the restore: {story.deleted}"
    assert not story.recalled_after_delete, f"the text is still recalled after the delete, before the restore: {story.recalled_after_delete}"
    assert not story.restore_error and story.restore.final.get("status") == "completed", (
        f"restore import: {story.restore_error or story.restore.final}"
    )
    assert not story.a_recalls_restored, f"A after the restore: {story.a_recalls_restored}"
    assert not story.b_blind_restored, f"B after the restore: {story.b_blind_restored}"


@proves("ingest.okf-bundle-import")
@negative_control
def test_restore_check_fails_for_the_user_the_record_was_never_shared_with(a, b, story):
    check_recalled(a, story.new, story.q_new)  # the same check passes for the owner first
    expect_check_fails(check_recalled, b, story.new, story.q_new)


@proves("archive.phase2-led-a", "archive.phase2-led-b", "contracts.lifecycle")
@expected_red(RED_DURABLE_WRITES)
def test_the_accepted_write_and_its_receipt_survive_a_restart_of_the_api(story):
    assert not story.restart_error, story.restart_error
    st = story.status_after_restart
    assert st.status == 200 and isinstance(st.body, dict), st.describe()
    assert story.save.task_id.startswith("intent-"), f"the write was accepted as {story.save.task_id!r}, not as a durable intent"
    assert st.body.get("status") == "completed" and receipt_of(st.body) == receipt_of(story.save.final), (
        f"receipt after the restart differs: {str(st.body)[:200]}"
    )


@proves("archive.phase2-port", "contracts.portability")
@expected_red(RED_PORTABILITY)
def test_the_tenant_export_is_a_verified_bundle_only_its_owner_can_download(b, story):
    p = story.portable
    assert p.status == 201 and isinstance(p.body, dict), p.describe()
    body = p.body
    assert body.get("export_id") and body.get("digest") and body.get("download_capability"), f"export answer: {sorted(body)}"
    assert story.portable_download.status == 200, story.portable_download.describe()
    assert story.portable_download.final.get("bytes", 0) > 0, "downloaded bundle is empty"
    other = b.get(f"/v1/portability/exports/{body['export_id']}/download",
                  headers={"X-Export-Capability": str(body["download_capability"])})
    assert other.status_code in DENIED, f"B downloaded A's export with A's capability -> {other.status_code}"


@proves("archive.phase2-port", "contracts.portability")
@negative_control
@expected_red(RED_PORTABILITY)
def test_portable_export_check_fails_for_a_user_without_the_right_to_export_the_tenant(a, b):
    def check_exported(client):
        r = client.post("/v1/portability/exports")
        assert r.status_code == 201 and r.json().get("digest"), f"POST /v1/portability/exports -> {r.status_code}"

    check_exported(a)  # the owner's export succeeds first
    expect_check_fails(check_exported, b)
