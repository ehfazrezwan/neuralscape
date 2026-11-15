"""Ingest: text, files, zips, OKF bundles, artifact storage, document extraction.

Runs on the dedicated ingest worker. Fact distillation (an LLM call) is switched
off (``extract_facts=false``) everywhere except the one text ingest that proves
the passage + fact pipeline; everything else asserts verbatim passages found by
a unique marker. The Docling scenario runs only under the ``docling`` profile.
"""

from __future__ import annotations

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, profiles, proves
from nsaccept.client import NSClient
from nsaccept.helpers import mcp_json
from nsaccept.pdfgen import make_pdf
from nsaccept.upload import build_zip, upload_files

INGEST_WAIT = 180


# ── the real checks ──────────────────────────────────────────────


def check_recalled(client: NSClient, marker: str, query: str, project_id: str) -> dict:
    results = client.search(query, project_id=project_id, limit=20)
    hits = [r for r in results if marker in r.get("memory", "")]
    assert hits, f"{marker} not in {len(results)} results for {query!r}"
    return hits[0]


def check_artifact_bytes(client: NSClient, file_id: str, expected: bytes) -> None:
    r = client.get(f"/v1/ingest/artifacts/{file_id}")
    assert r.status_code == 200, f"GET artifact -> {r.status_code}: {r.text[:120]}"
    assert r.content == expected, "downloaded bytes differ from the uploaded file"


def poll_ok(client: NSClient, task_id: str, timeout: float = INGEST_WAIT) -> dict:
    status = client.poll_task(task_id, timeout=timeout)
    assert status.get("status") == "completed", f"ingest task ended as {status}"
    result = status.get("result") or {}
    assert not result.get("skipped"), f"ingest skipped the input: {result.get('reason')}"
    return result


# ── module fixtures ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def text_ingest(ctx, a) -> dict:
    """The one ingest that runs fact distillation (LLM) next to passage indexing."""
    marker = f"zq-{ctx.run_id}-ingtext"
    note = (
        f"# Pager rotation notes\n\n{marker} The incident pager rotation is called Heron and it rotates "
        f"every Monday morning.\n\nThe escalation contact for the Heron rotation is the harbour desk.\n"
    )
    out: dict = {"marker": marker, "query": f"{marker} incident pager rotation Heron", "error": ""}
    try:
        r = a.post("/v1/ingest/text", json={
            "content": note, "title": f"{marker}-note", "category": "tech_stack", "scope": "project",
            "project_id": ctx.project_id, "visibility": "private", "extract_facts": True,
        })
        assert r.status_code == 202, f"POST /v1/ingest/text -> {r.status_code}: {r.text[:200]}"
        out["task_id"] = r.json()["task_id"]
        out["result"] = poll_ok(a, out["task_id"])
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


@pytest.fixture(scope="module")
def upload(ctx, a) -> dict:
    marker = f"zq-{ctx.run_id}-ingfile"
    body = (
        f"# Mooring log\n\n{marker} The mooring ropes are replaced every spring by the dock crew.\n"
    ).encode()
    out: dict = {"marker": marker, "body": body, "name": f"{marker}.md", "error": ""}
    try:
        r = upload_files(
            a, {out["name"]: body}, mime="text/markdown", category="tech_stack", scope="project",
            project_id=ctx.project_id, visibility="private", extract_facts=False,
        )
        info = r.json()
        assert info["count"] == 1 and info["files"][0]["filename"] == out["name"], f"upload response: {info}"
        out["file_id"] = info["files"][0]["file_id"]
        out["task_id"] = info["files"][0]["task_id"]
        out["result"] = poll_ok(a, out["task_id"])
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


@pytest.fixture(scope="module")
def okf(ctx, a) -> dict:
    marker = f"zq-{ctx.run_id}-ingokf"
    bundle = build_zip({
        "index.md": "---\nokf_version: 0.1\n---\n",
        "runbooks/mooring.md": (
            "---\ntype: Guide\ntitle: Mooring runbook\n---\n"
            f"# Mooring runbook\n\n{marker} Tie off at the starboard cleat before the tide turns.\n"
        ),
    })
    out: dict = {"marker": marker, "error": ""}
    try:
        r = upload_files(
            a, {f"{marker}.zip": bundle}, mime="application/zip", category="tech_stack", scope="project",
            project_id=ctx.project_id, visibility="private", extract_facts=False,
        )
        info = r.json()
        assert info["count"] == 1, f"a single OKF bundle should be one job, got {info}"
        out["task_id"] = info["files"][0]["task_id"]
        out["result"] = poll_ok(a, out["task_id"])
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


# ── text ingest ──────────────────────────────────────────────────


@proves("ingest.text-and-document", "ingest.pipeline", "ingest.artifact-storage")
def test_text_ingest_produces_passages_and_distilled_facts_that_are_recalled(ctx, a, text_ingest):
    assert not text_ingest["error"], text_ingest["error"]
    result = text_ingest["result"]
    assert result["passages"] >= 1, f"no passages indexed: {result}"
    assert result["facts"] >= 1, f"no facts distilled from a text with explicit statements: {result}"
    hit = eventually(
        lambda: check_recalled(a, text_ingest["marker"], text_ingest["query"], ctx.project_id),
        timeout=90, what="ingested text recalled",
    )
    assert hit.get("source_ref") or hit.get("memory_kind") == "passage", f"recalled row carries no provenance: {hit}"


@proves("ingest.text-and-document", "ingest.pipeline")
@negative_control
def test_text_recall_check_fails_for_user_b(ctx, b, text_ingest):
    assert not text_ingest["error"], text_ingest["error"]
    expect_check_fails(check_recalled, b, text_ingest["marker"], text_ingest["query"], ctx.project_id)


@proves("ingest.text-and-document")
def test_mcp_ingest_text_waits_for_completion(ctx, a, mcp_a):
    mcp_a.initialize()
    marker = f"zq-{ctx.run_id}-mcping"
    out = mcp_json(mcp_a, "ingest_text", {
        "content": f"{marker} The foghorn schedule is posted on the east wall of the signal house.",
        "title": f"{marker}-note", "category": "tech_stack", "project_id": ctx.project_id,
        "visibility": "private", "extract_facts": False, "wait": True,
    })
    assert out.get("status") == "completed", f"MCP ingest_text: {str(out)[:240]}"
    assert out["result"]["passages"] >= 1, f"MCP ingest result: {out['result']}"
    eventually(
        lambda: check_recalled(a, marker, f"{marker} foghorn schedule signal house", ctx.project_id),
        timeout=90, what="MCP-ingested text recalled",
    )


@proves("ingest.text-and-document")
def test_ingest_text_validation_refuses_a_project_scope_without_a_project(a):
    r = a.post("/v1/ingest/text", json={"content": "x", "scope": "project"})
    assert r.status_code == 400, f"scope=project without project_id -> {r.status_code}"


# ── file upload + artifact storage ───────────────────────────────


@proves("ingest.files-upload", "ingest.pipeline", "ingest.artifact-storage")
def test_uploaded_markdown_file_is_ingested_and_recalled(ctx, a, upload):
    assert not upload["error"], upload["error"]
    assert upload["result"]["passages"] >= 1, f"upload result: {upload['result']}"
    assert upload["result"]["doc_type"] == "plain", f"markdown should need no extractor: {upload['result']['doc_type']}"
    eventually(
        lambda: check_recalled(a, upload["marker"], f"{upload['marker']} mooring ropes dock crew", ctx.project_id),
        timeout=90, what="uploaded file recalled",
    )


@proves("ingest.files-upload")
def test_zip_upload_is_expanded_into_one_job_per_member(ctx, a):
    marker = f"zq-{ctx.run_id}-ingzip"
    archive = build_zip({
        "notes/one.md": f"{marker}-one The ferry berth is numbered four.\n",
        "notes/two.md": f"{marker}-two The ferry berth lights are amber.\n",
    })
    r = upload_files(
        a, {f"{marker}.zip": archive}, mime="application/zip", category="tech_stack", scope="project",
        project_id=ctx.project_id, visibility="private", extract_facts=False,
    )
    info = r.json()
    assert info["count"] == 2, f"zip of two files made {info['count']} jobs: {info}"
    for f in info["files"]:
        assert poll_ok(a, f["task_id"])["passages"] >= 1
    for suffix, query in (("-one", "ferry berth numbered four"), ("-two", "ferry berth lights amber")):
        eventually(lambda s=suffix, q=query: check_recalled(a, marker + s, f"{marker}{s} {q}", ctx.project_id), timeout=90, what=f"zip member {suffix}")


@proves("ingest.files-upload")
def test_upload_validation_refuses_bad_forms(ctx, a):
    body = {"x.md": b"x"}
    assert upload_files(a, body, expect=None, category="nonsense").status_code == 400, "bad category accepted"
    assert upload_files(a, body, expect=None, scope="project").status_code == 400, "project scope without project_id accepted"
    assert upload_files(a, body, expect=None, adapter="no such adapter!").status_code == 400, "bad adapter accepted"


@proves("ingest.artifact-storage", "ingest.files-upload")
def test_artifact_download_is_byte_exact_for_the_owner_and_hidden_from_others(a, b, upload):
    assert not upload["error"], upload["error"]
    check_artifact_bytes(a, upload["file_id"], upload["body"])
    r = b.get(f"/v1/ingest/artifacts/{upload['file_id']}")
    assert r.status_code in (403, 404), f"B fetching A's artifact -> {r.status_code}"
    assert upload["marker"] not in r.text, "refusal leaks the artifact"
    assert a.get("/v1/ingest/artifacts/not-a-hash").status_code == 404, "malformed artifact id is not a 404"


@proves("ingest.artifact-storage")
@negative_control
def test_artifact_check_fails_for_user_b(b, upload):
    assert not upload["error"], upload["error"]
    expect_check_fails(check_artifact_bytes, b, upload["file_id"], upload["body"])


# ── OKF bundle import ────────────────────────────────────────────


@proves("ingest.okf-bundle-import", "ingest.files-upload")
def test_okf_bundle_zip_is_imported_as_one_knowledge_bundle(ctx, a, okf):
    assert not okf["error"], okf["error"]
    assert okf["result"]["concepts"] == 1 and okf["result"]["passages"] >= 1, f"OKF import result: {okf['result']}"
    hit = eventually(
        lambda: check_recalled(a, okf["marker"], f"{okf['marker']} tie off starboard cleat tide", ctx.project_id),
        timeout=90, what="OKF concept recalled",
    )
    assert (hit.get("source_ref") or {}).get("connector_type") == "okf_bundle", f"source_ref: {hit.get('source_ref')}"


@proves("ingest.okf-bundle-import")
@negative_control
def test_okf_recall_check_fails_for_user_b(ctx, b, okf):
    assert not okf["error"], okf["error"]
    expect_check_fails(check_recalled, b, okf["marker"], f"{okf['marker']} tie off starboard cleat tide", ctx.project_id)


# ── document extraction (Docling profile only) ───────────────────


@proves("ingest.extract-docling", "ingest.files-upload")
@profiles("docling")
@needs("docling")
def test_pdf_is_extracted_by_docling_and_its_text_recalled(ctx, a):
    marker = f"zq-{ctx.run_id}-ingpdf"
    pdf = make_pdf([f"{marker} Tidal Almanac", "The almanac lists the lunar tide tables for the harbour."])
    r = upload_files(
        a, {f"{marker}.pdf": pdf}, mime="application/pdf", category="tech_stack", scope="project",
        project_id=ctx.project_id, visibility="private", extract_facts=False,
    )
    result = poll_ok(a, r.json()["files"][0]["task_id"], timeout=300)
    assert result["doc_type"] == "docling", f"PDF was parsed by {result['doc_type']!r}, not docling"
    assert result["passages"] >= 1, f"PDF yielded no passages: {result}"
    eventually(
        lambda: check_recalled(a, marker, f"{marker} tidal almanac lunar tide tables harbour", ctx.project_id),
        timeout=90, what="PDF text recalled",
    )


@proves("ingest.extract-docling")
@profiles("docling")
@negative_control
@needs("docling")
def test_pdf_doc_type_check_fails_for_a_plain_text_upload(ctx, a):
    marker = f"zq-{ctx.run_id}-ingplain"
    r = upload_files(
        a, {f"{marker}.md": f"{marker} plain text\n".encode()}, mime="text/markdown", category="tech_stack",
        scope="project", project_id=ctx.project_id, visibility="private", extract_facts=False,
    )
    result = poll_ok(a, r.json()["files"][0]["task_id"])

    def check_docling():
        assert result["doc_type"] == "docling", f"parsed by {result['doc_type']!r}"

    expect_check_fails(check_docling)
