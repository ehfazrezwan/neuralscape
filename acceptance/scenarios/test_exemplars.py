"""Visual setup exemplars (profile ``docling``: EXEMPLAR_STORE_ENABLED + DOCLING_EXTRACT_IMAGES).

A generated PDF with an embedded figure is uploaded under the ``trading_strategy`` adapter; the
ingest worker has Docling hand back the figure, stores its bytes in the exemplar store and writes
a ``setup`` memory tagged ``visual_exemplar`` whose source_ref points at the image. The image is
then served by ``GET /v1/ingest/exemplars/{id}`` to its owner and to nobody else.

The vision description needs the external LLM gateway, which this stack does not have; the
service stores the image and a minimal memory without it, and that is what is asserted here.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from nsaccept import expect_check_fails, needs, negative_control, profiles, proves
from nsaccept.client import NSClient
from nsaccept.pdfgen import make_pdf_with_image
from nsaccept.upload import upload_files

pytestmark = [profiles("docling"), needs("docling")]

INGEST_WAIT = 600
IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF")


def exemplar_id_from(summary: dict) -> str:
    """The image id (``source_ref.external_id``) is the first 16 hex of the stored file's name."""
    uri = summary.get("image_uri") or ""
    stem = urlparse(uri).path.rsplit("/", 1)[-1].split(".")[0]
    assert len(stem) >= 16, f"exemplar summary has no image_uri to derive the id from: {summary}"
    return stem[:16]


def check_serves_image(client: NSClient, image_id: str) -> bytes:
    r = client.get(f"/v1/ingest/exemplars/{image_id}")
    assert r.status_code == 200, f"GET /v1/ingest/exemplars/{image_id} as {client.user_id} -> {r.status_code}: {r.text[:120]}"
    assert r.headers["content-type"].startswith("image/"), f"content-type {r.headers['content-type']!r}"
    assert r.content.startswith(IMAGE_MAGIC), f"served bytes are not an image: {r.content[:8]!r}"
    return r.content


@pytest.fixture(scope="module")
def exemplar(ctx, a) -> dict:
    marker = f"zq-{ctx.run_id}-exemplar"
    out: dict = {"marker": marker, "error": ""}
    try:
        pdf = make_pdf_with_image([f"{marker} Kangaroo tail setup example.", "A long tail rejects the support zone."])
        r = upload_files(
            a, {f"{marker}.pdf": pdf}, mime="application/pdf", adapter="trading_strategy", scope="project",
            project_id=ctx.project_id, category="setup", visibility="private", extract_facts=False,
            tags=f"strategy:acc{ctx.run_id}",
        )
        task_id = r.json()["files"][0]["task_id"]
        st = a.poll_task(task_id, timeout=INGEST_WAIT)
        assert st.get("status") == "completed", f"ingest task ended as {st}"
        result = st.get("result") or {}
        assert not result.get("skipped"), f"ingest skipped the PDF: {result.get('reason')}"
        out["result"] = result
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


@proves("ingest.visual-exemplars")
def test_a_figure_in_a_pdf_becomes_a_stored_exemplar_memory(ctx, a, exemplar):
    assert not exemplar["error"], exemplar["error"]
    summaries = exemplar["result"].get("exemplars") or []
    assert summaries, f"Docling handed back no figure to index (ingest result keys: {sorted(exemplar['result'])})"
    exemplar["id"] = exemplar_id_from(summaries[0])
    memory_id = (summaries[0].get("memory_ids") or [""])[0]
    assert memory_id, f"exemplar summary has no memory id: {summaries[0]}"
    row = a.get(f"/v1/memories/{memory_id}")
    assert row.status_code == 200, f"GET /v1/memories/{memory_id} -> {row.status_code}"
    body = row.json()
    assert body.get("category") == "setup" and "visual_exemplar" in (body.get("tags") or []), f"exemplar memory: category {body.get('category')!r} tags {body.get('tags')}"
    assert f"strategy:acc{ctx.run_id}" in (body.get("tags") or []), "the strategy tag was not carried onto the exemplar"
    ref = body.get("source_ref") or {}
    assert ref.get("external_id") == exemplar["id"] and ref.get("stored_path", "").startswith("file://"), f"source_ref: {ref}"


@proves("ingest.visual-exemplars")
def test_the_exemplar_image_is_served_to_its_owner_only(ctx, a, b, exemplar):
    assert not exemplar["error"], exemplar["error"]
    summaries = exemplar["result"].get("exemplars") or []
    assert summaries, "no exemplar was indexed"
    image_id = exemplar_id_from(summaries[0])
    data = check_serves_image(a, image_id)
    assert len(data) > 100, f"served image is {len(data)} bytes"
    assert b.get(f"/v1/ingest/exemplars/{image_id}").status_code == 404, "B can fetch A's exemplar image"
    assert a.get("/v1/ingest/exemplars/0000000000000000").status_code == 404, "an unknown image id did not 404"


@proves("ingest.visual-exemplars")
@negative_control
def test_image_check_fails_for_another_user_and_for_a_made_up_id(a, b, exemplar):
    assert not exemplar["error"], exemplar["error"]
    summaries = exemplar["result"].get("exemplars") or []
    assert summaries, "no exemplar was indexed"
    image_id = exemplar_id_from(summaries[0])
    expect_check_fails(check_serves_image, b, image_id)
    expect_check_fails(check_serves_image, a, "0" * 16)
