"""Ingest Graphify output bundles (``graph.json`` + ``GRAPH_REPORT.md``).

A Graphify run leaves a ``graphify-out/`` directory. Users upload it to
``POST /v1/ingest/files`` either as a zip (expanded by the existing
``ingest.archive`` guards) or as individual files; each member arrives here as
its own ingest-worker job. Detection is keyed off the two canonical filenames:

- ``graph.json`` (a node-link graph with ``nodes`` + ``links``/``edges``) —
  routed to :func:`ingest_code_graph_json`, which distils ONLY the stable
  semantic layer (communities, god nodes, surprising connections, rationale
  nodes — see :mod:`adapters.code_graph.semantic`) into memories. The raw
  graph is never mirrored: it churns with every commit and stays queryable
  live through NS's ``query_code_graph`` tools.
- ``GRAPH_REPORT.md`` — flows through the NORMAL text pipeline, upgraded to
  the ``code_graph`` adapter (section-aware chunking + the report insight
  extractor) when the caller didn't explicitly pick another adapter.

Every produced memory's ``source_ref`` resolves through **NS's own surface**:
``url`` points at ``/v1/code-graph/query`` and the ``retrieval`` handle names
NS's ``query_code_graph`` MCP tool with the bundle's ``graph_id`` (the stored
artifact id) — never Graphify's own MCP server.

Import-safe without the optional graphifyy library; the graphify-dependent
work is imported lazily behind :func:`adapters.code_graph.code_graph_available`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from urllib.parse import quote

from config import settings
from ingest.pipeline import _fact_scope

logger = logging.getLogger(__name__)

GRAPH_JSON_BASENAME = "graph.json"
GRAPH_REPORT_BASENAME = "graph_report.md"  # matched casefolded

# Only sniff JSON structure on files that could plausibly be a code graph
# (uploads are already capped at ingest_max_file_mb; this is defense-in-depth
# for direct callers).
_MAX_SNIFF_BYTES = 64 * 1024 * 1024


def detect_graphify_member(filename: str, data: bytes) -> str | None:
    """Classify an uploaded file as part of a Graphify bundle.

    Returns ``"graph"`` for a graph.json (name + node-link JSON shape),
    ``"report"`` for a GRAPH_REPORT.md (name), else ``None``.
    """
    base = os.path.basename(filename or "").casefold()
    if base == GRAPH_REPORT_BASENAME:
        return "report"
    if base != GRAPH_JSON_BASENAME:
        return None
    if len(data) > _MAX_SNIFF_BYTES:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
        return None
    links = payload.get("links", payload.get("edges"))
    if not isinstance(links, list):
        return None
    return "graph"


def _code_graph_source_ref(base_ref: dict, graph_id: str | None, fact) -> dict:
    """Build one semantic fact's source_ref — resolving through NS's endpoint.

    ``base_ref`` is the stored artifact's descriptor (title/url/external_id);
    we keep the artifact download handle reachable via ``parent_id`` and point
    ``url`` + ``retrieval`` at NS's code-graph query surface with the node id.
    """
    ref = dict(base_ref)
    ref.update(
        {
            "connector_id": "graphify",
            "connector_type": "code_graph",
            # The node/community id inside graph.json this memory came from.
            "external_id": fact.external_id,
            "title": fact.title,
        }
    )
    if graph_id:
        # All memories of one bundle share the artifact id as parent.
        ref["parent_id"] = graph_id
        q = quote(fact.title, safe="")
        ref["url"] = f"/v1/code-graph/query?graph_id={graph_id}&question={q}"
        ref["retrieval"] = {
            "mcp_server": "neuralscape",
            "tool": "query_code_graph",
            "args": {"graph_id": graph_id, "question": fact.title},
        }
    return ref


def ingest_code_graph_json(service, data: bytes, payload: dict) -> dict:
    """Distil a graph.json's stable semantic layer into memories.

    Runs on the ingest worker (sync, called via ``asyncio.to_thread``).
    ``payload`` is the ingest-file job payload (``source_ref``, ``user_id``,
    ``options``). Returns the same summary shape as
    :func:`ingest.pipeline.ingest_document` (including deferred ``graph_jobs``
    the worker enqueues with the ``code_graph`` ontology).

    Idempotent like every ingest: ``store_raw`` dedupes on content hash, and
    dedup hits produce no graph job.
    """
    from adapters.code_graph.semantic import extract_semantic_layer, load_code_graph

    options = payload.get("options", {})
    user_id = payload["user_id"]
    base_ref = dict(payload.get("source_ref") or {})
    # The stored artifact's id doubles as the bundle's graph_id — the handle
    # the query tools resolve (owner-scoped) back to this exact graph.json.
    graph_id = base_ref.get("external_id")

    # graphify's loader reads from a path; the job hands us bytes. Materialize
    # to a temp file (small compared to a Docling parse; keeps us on the
    # library's loader instead of hand-rolled parsing).
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        G = load_code_graph(tmp.name)
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    facts = extract_semantic_layer(G, settings)

    project_id = options.get("project_id")
    base_tags = options.get("tags") or []
    memory_ids: list[str] = []
    graph_jobs: list[dict] = []
    stored_count = 0
    for fact in facts:
        scope_val, fact_pid = _fact_scope(fact.category, project_id)
        source_ref = _code_graph_source_ref(base_ref, graph_id, fact)
        tags = list(dict.fromkeys([*base_tags, *fact.tags])) or None
        try:
            stored, created = service.store_raw(
                content=fact.content,
                user_id=user_id,
                category=fact.category,
                scope=scope_val,
                project_id=fact_pid,
                tags=tags,
                agent_id=options.get("agent_id"),
                run_id=options.get("run_id"),
                source_type="imported",
                epistemic_level=fact.epistemic_level,
                confidence=fact.confidence,
                visibility=options.get("visibility"),
                memory_kind="fact",
                source_ref=source_ref,
                add_to_graph=False,
                return_created=True,
            )
            memory_ids.extend(m.id for m in stored)
            stored_count += len(stored)
            if created:
                for m in stored:
                    graph_jobs.append(
                        {
                            "memory_id": m.id,
                            "content": fact.content,
                            "user_id": user_id,
                            "project_id": fact_pid,
                            "visibility": getattr(m, "visibility", None),
                            "source_ref": source_ref,
                        }
                    )
        except Exception as e:  # noqa: BLE001 — one bad fact must not sink the bundle
            logger.warning("Code-graph fact store failed (%s): %s", fact.category, e)

    logger.info(
        "Ingested code graph graph_id=%s → %d semantic facts "
        "(%d graph jobs deferred; %d nodes / %d edges left live in graph.json)",
        graph_id, stored_count, len(graph_jobs),
        G.number_of_nodes(), G.number_of_edges(),
    )
    return {
        "passages": 0,
        "facts": stored_count,
        "memory_ids": memory_ids,
        "parent_id": graph_id,
        "graph_jobs": graph_jobs,
        "adapter": "code_graph",
        "graph_id": graph_id,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }
