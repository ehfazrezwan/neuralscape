"""An ingest request's ``workspace`` reaches the job and every stored row, on every ingest surface."""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import worker
from config import settings
from schemas import MemoryResponse
from tokens import issue_user_token

SECRET = "x" * 32


@pytest.fixture
def tm(monkeypatch):
    manager = MagicMock()
    for name in ("enqueue_ingest_document", "enqueue_ingest_file", "enqueue_ingest_okf_bundle"):
        setattr(manager, name, AsyncMock(return_value="task-1"))
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "ingest_storage_enabled", False)
    return manager


def _headers():
    return {"Authorization": f"Bearer {issue_user_token('alice', SECRET, 3600)}"}


def _client():
    return TestClient(main.app)


# ── REST ──


def test_rest_ingest_document_forwards_workspace(tm):
    body = {
        "content": "doc text",
        "source": {"connector_id": "c", "connector_type": "manual", "external_id": "e1"},
        "workspace": "ref-books",
    }
    assert _client().post("/v1/ingest", json=body, headers=_headers()).status_code == 202
    assert tm.enqueue_ingest_document.call_args.args[0]["workspace"] == "ref-books"


def test_rest_ingest_text_accepts_and_forwards_workspace(tm):
    resp = _client().post("/v1/ingest/text", json={"content": "notes", "workspace": "ref-books"}, headers=_headers())
    assert resp.status_code == 202, resp.text
    assert tm.enqueue_ingest_document.call_args.args[0]["workspace"] == "ref-books"


def test_rest_ingest_text_rejects_a_malformed_workspace(tm):
    resp = _client().post("/v1/ingest/text", json={"content": "notes", "workspace": "bad name!"}, headers=_headers())
    assert resp.status_code == 422


def test_rest_ingest_files_forwards_workspace_in_the_job_options(tm):
    files = {"files": ("a.md", b"# hello\n\nbody", "text/markdown")}
    resp = _client().post("/v1/ingest/files", files=files, data={"workspace": "ref-books"}, headers=_headers())
    assert resp.status_code == 202, resp.text
    assert tm.enqueue_ingest_file.call_args.args[0]["options"]["workspace"] == "ref-books"


def test_rest_ingest_files_rejects_a_malformed_workspace(tm):
    files = {"files": ("a.md", b"# hello", "text/markdown")}
    resp = _client().post("/v1/ingest/files", files=files, data={"workspace": "bad name!"}, headers=_headers())
    assert resp.status_code == 400


def test_rest_ingest_files_without_workspace_adds_no_option(tm):
    files = {"files": ("a.md", b"# hello\n\nbody", "text/markdown")}
    _client().post("/v1/ingest/files", files=files, headers=_headers())
    assert "workspace" not in tm.enqueue_ingest_file.call_args.args[0]["options"]


# ── MCP ──


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", [
    ("ingest_text", {"content": "notes"}),
    ("ingest_document", {"content": "doc", "source": {"connector_id": "c", "connector_type": "manual", "external_id": "e"}}),
])
async def test_mcp_ingest_tools_forward_workspace(monkeypatch, tool, args):
    import mcp_server

    manager = MagicMock()
    manager.enqueue_ingest_document = AsyncMock(return_value="task-1")
    monkeypatch.setattr(mcp_server, "_task_manager", manager)
    monkeypatch.setattr(settings, "ingest_storage_enabled", False)
    out = await mcp_server.call_tool(tool, {**args, "user_id": "alice", "workspace": "ref-books"})
    assert json.loads(out[0].text)["status"] == "accepted", out[0].text
    assert manager.enqueue_ingest_document.call_args.args[0]["workspace"] == "ref-books"


@pytest.mark.asyncio
async def test_mcp_ingest_document_rejects_a_malformed_workspace(monkeypatch):
    import mcp_server

    manager = MagicMock()
    manager.enqueue_ingest_document = AsyncMock(return_value="task-1")
    monkeypatch.setattr(mcp_server, "_task_manager", manager)
    out = await mcp_server.call_tool(
        "ingest_document",
        {"content": "d", "source": {"connector_id": "c", "connector_type": "manual", "external_id": "e"},
         "user_id": "alice", "workspace": "bad name!"},
    )
    assert json.loads(out[0].text)["status"] == "error"
    manager.enqueue_ingest_document.assert_not_called()


# ── job identity ──


@pytest.mark.asyncio
async def test_same_content_in_another_workspace_is_a_distinct_job():
    from task_manager import TaskManager

    tm = TaskManager()
    tm.pool = MagicMock()
    tm.pool.enqueue_job = AsyncMock(return_value=None)
    base = {"content": "same", "source": {"connector_id": "c"}, "user_id": "alice"}
    await tm.enqueue_ingest_document(base)
    await tm.enqueue_ingest_document({**base, "workspace": "ref-a"})
    await tm.enqueue_ingest_document({**base, "workspace": "ref-b"})
    file_payload = {"filename": "a.md", "user_id": "alice", "source_ref": {"external_id": "h"}, "options": {}}
    await tm.enqueue_ingest_file(file_payload)
    await tm.enqueue_ingest_file({**file_payload, "options": {"workspace": "ref-a"}})
    await tm.enqueue_ingest_okf_bundle(file_payload)
    await tm.enqueue_ingest_okf_bundle({**file_payload, "options": {"workspace": "ref-a"}})
    ids = [c.kwargs["_job_id"] for c in tm.pool.enqueue_job.call_args_list]
    assert len(set(ids[0:3])) == 3 and len(set(ids[3:5])) == 2 and len(set(ids[5:7])) == 2


# ── worker jobs ──


@pytest.fixture
def ctx():
    redis = MagicMock()
    redis.enqueue_job = AsyncMock()
    return {"service": MagicMock(), "redis": redis}


@pytest.mark.asyncio
async def test_document_job_builds_the_ingest_doc_with_workspace(ctx, monkeypatch):
    seen = {}
    monkeypatch.setattr("ingest.pipeline.ingest_document", lambda s, d: seen.setdefault("d", d) and {"graph_jobs": []})
    await worker.process_ingest_document(
        ctx, {"content": "x", "source": {}, "user_id": "alice", "workspace": "ref-books"}
    )
    assert seen["d"].workspace == "ref-books"


@pytest.mark.asyncio
async def test_file_job_builds_the_ingest_doc_with_workspace(ctx, monkeypatch):
    seen = {}
    monkeypatch.setattr("ingest.pipeline.ingest_document", lambda s, d: seen.setdefault("d", d) and {"graph_jobs": []})
    monkeypatch.setattr("ingest.extract.extract_text", lambda *a, **k: ("# text", "markdown"), raising=False)
    import ingest.extract as extract_mod

    monkeypatch.setattr(extract_mod, "extract_text", lambda *a, **k: ("# text", "markdown"))
    payload = {
        "filename": "a.md", "user_id": "alice", "data_b64": "IyB0ZXh0",
        "source_ref": {"external_id": "h"}, "options": {"workspace": "ref-books"},
    }
    await worker.process_ingest_file(ctx, payload)
    assert seen["d"].workspace == "ref-books"


@pytest.mark.asyncio
async def test_okf_job_passes_workspace_to_the_walker(ctx, monkeypatch):
    import ingest.okf_bundle as okf

    seen = {}
    monkeypatch.setattr(okf, "load_bundle_zip", lambda *a, **k: {"a.md": "x"})
    monkeypatch.setattr(okf, "ingest_okf_bundle", lambda service, **kw: seen.update(kw) or {"graph_jobs": [], "memory_ids": []})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.md", "x")
    import base64

    await worker.process_ingest_okf_bundle(ctx, {
        "filename": "b.zip", "user_id": "alice", "data_b64": base64.b64encode(buf.getvalue()).decode(),
        "source_ref": {}, "options": {"workspace": "ref-books"},
    })
    assert seen["workspace"] == "ref-books"


def test_okf_walker_applies_workspace_to_concepts_and_link_memories():
    from ingest.okf_bundle import ingest_okf_bundle

    service = MagicMock()
    service.store_raw.return_value = ([MemoryResponse(id="m", memory="x", category="domain_knowledge")], True)
    files = {
        "okf_version": "1",
        "a.md": "---\ntype: concept\nid: a\nlinks: [b]\n---\nAlpha text body.",
        "b.md": "---\ntype: concept\nid: b\n---\nBeta text body.",
    }
    ingest_okf_bundle(service, files=files, bundle_uri="x.zip", user_id="alice", extract_facts=False,
                      workspace="ref-books", llm_call=lambda p: "domain_knowledge")
    assert service.store_raw.call_count >= 2
    assert {c.kwargs.get("workspace") for c in service.store_raw.call_args_list} == {"ref-books"}


@pytest.mark.asyncio
async def test_connector_sync_uses_the_connectors_default_workspace(ctx, monkeypatch):
    from cryptography.fernet import Fernet

    from connectors.vault import ConnectorVault
    from tests.fake_vault_store import FakeVaultStore

    vault = ConnectorVault(FakeVaultStore(), Fernet(Fernet.generate_key()))
    await vault.put({
        "connector_id": "c1", "connector_type": "generic_rest", "owner_user_id": "alice", "credentials": {},
        "config": {"list_url": "http://x/items"}, "default_workspace": "ref-books",
    })
    gen = (await vault.get("c1"))["generation"]

    class A:
        async def list_resources(self, cursor=None):
            from connectors.base import ConnectorResource
            return [ConnectorResource(external_id="i1", revision="1")], None
        async def fetch(self, r): return "text"
        def source_descriptor(self, r): return {"external_id": r.external_id}
        async def aclose(self): pass

    seen = []
    monkeypatch.setattr("connectors.registry.build_adapter", lambda rec: A())
    monkeypatch.setattr("ingest.pipeline.ingest_document", lambda s, d: seen.append(d) or {"graph_jobs": []})
    ctx["vault"] = vault
    await worker.process_connector_sync(ctx, "c1", gen)
    assert [d.workspace for d in seen] == ["ref-books"]


# ── rows: passages AND facts ──


def test_pipeline_stores_passages_and_facts_in_the_workspace():
    from ingest.pipeline import IngestDoc, ingest_document

    service = MagicMock()
    service.store_raw.side_effect = lambda **kw: (
        ([MemoryResponse(id="m", memory="x", category="domain_knowledge")], True) if kw.get("return_created")
        else [MemoryResponse(id="p", memory="x", category="domain_knowledge")]
    )
    service.extract_facts_only.return_value = [("domain_knowledge", "A distilled fact")]
    doc = IngestDoc(content="# T\n\nSome passage text here.", source={"connector_id": "c", "external_id": "e"},
                    user_id="alice", workspace="ref-books")
    out = ingest_document(service, doc)
    kinds = {c.kwargs["memory_kind"]: c.kwargs["workspace"] for c in service.store_raw.call_args_list}
    assert kinds == {"passage": "ref-books", "fact": "ref-books"}, out


# ── review follow-ups: normalisation, shared validator, code-graph path ──


def test_normalize_workspace_is_the_single_rule():
    from schemas import normalize_workspace

    assert normalize_workspace(None) is None and normalize_workspace("") is None and normalize_workspace("memory") is None
    assert normalize_workspace("ref-books") == "ref-books"
    for bad in ("bad name!", "x" * 101, "a/b"):
        with pytest.raises(ValueError):
            normalize_workspace(bad)


def test_pipeline_stores_an_explicit_memory_workspace_untagged_and_not_the_adapter_default():
    from ingest.pipeline import IngestDoc, ingest_document

    for explicit, adapter in (("memory", "default"), ("memory", "trading_strategy"), ("", "default")):
        service = MagicMock()
        service.store_raw.return_value = ([MemoryResponse(id="p", memory="x", category="domain_knowledge")], True)
        service.extract_facts_only.return_value = []
        ingest_document(service, IngestDoc(content="# T\n\nSome text for a passage.", source={"connector_id": "c", "external_id": "e"},
                                           user_id="alice", workspace=explicit, adapter=adapter))
        assert {c.kwargs["workspace"] for c in service.store_raw.call_args_list} == {None}, (explicit, adapter)


def test_a_non_default_adapter_without_a_workspace_still_gets_its_reference_workspace():
    from ingest.pipeline import IngestDoc, ingest_document

    service = MagicMock()
    service.store_raw.return_value = ([MemoryResponse(id="p", memory="x", category="domain_knowledge")], True)
    service.extract_facts_only.return_value = []
    ingest_document(service, IngestDoc(content="# T\n\nSome text for a passage.", source={"connector_id": "c", "external_id": "e"},
                                       user_id="alice", adapter="trading_strategy"))
    assert {c.kwargs["workspace"] for c in service.store_raw.call_args_list} == {"ref-trading_strategy"}


def test_rest_routes_normalise_an_explicit_memory_workspace(tm):
    _client().post("/v1/ingest/text", json={"content": "notes", "workspace": "memory"}, headers=_headers())
    assert "workspace" not in tm.enqueue_ingest_document.call_args.args[0]
    files = {"files": ("a.md", b"# hello\n\nbody", "text/markdown")}
    _client().post("/v1/ingest/files", files=files, data={"workspace": "memory"}, headers=_headers())
    assert "workspace" not in tm.enqueue_ingest_file.call_args.args[0]["options"]


def test_code_graph_ingest_stores_facts_in_the_requested_workspace(monkeypatch):
    import ingest.code_graph as cg
    from types import SimpleNamespace

    fact = SimpleNamespace(content="module a imports b", category="architecture", tags=[], epistemic_level="explicit", confidence=0.9)
    monkeypatch.setattr("adapters.code_graph.semantic.load_code_graph", lambda path: MagicMock(number_of_nodes=lambda: 1, number_of_edges=lambda: 0), raising=False)
    monkeypatch.setattr("adapters.code_graph.semantic.extract_semantic_layer", lambda G, s: [fact], raising=False)
    monkeypatch.setattr(cg, "_code_graph_source_ref", lambda base, gid, f: {"external_id": "e"})
    service = MagicMock()
    service.store_raw.return_value = ([MemoryResponse(id="m", memory="x", category="architecture")], True)
    payload = {"user_id": "alice", "source_ref": {"external_id": "g"}, "options": {"workspace": "ref-code"}}
    cg.ingest_code_graph_json(service, b"{}", payload)
    assert service.store_raw.call_args.kwargs["workspace"] == "ref-code"
    payload["options"]["workspace"] = "memory"
    cg.ingest_code_graph_json(service, b"{}", payload)
    assert service.store_raw.call_args.kwargs["workspace"] is None


def test_the_workspace_pattern_has_one_definition():
    import schemas

    assert schemas.RawMemoryRequest.model_fields["workspace"].metadata
    patterns = {m.pattern for f in (schemas.RawMemoryRequest, schemas.IngestDocumentRequest, schemas.IngestTextRequest,
                                     schemas.ConnectorConfigRequest)
                for name in ("workspace", "default_workspace") if name in f.model_fields
                for m in f.model_fields[name].metadata if hasattr(m, "pattern")}
    assert patterns == {schemas.WORKSPACE_PATTERN}
