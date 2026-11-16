"""Edit path: patch_memory and bulk retag housekeeping.

Mechanically extracted from memory_service.py (mixins-with-facade split);
the revision precondition and the store-level compare-and-set were added on top.

GUARANTEE (what correctness rests on): the memory's TEXT, metadata and REVISION
change together in ONE filter-scoped ``set_payload`` whose filter is
``id == <memory> AND metadata.revision == <expected>``. Qdrant applies it only to
a point that exists and still matches, so a stale writer cannot overwrite a later
edit and a deleted point cannot be re-created. Whether it applied is read back:
the stored revision map holds OUR writer token for the revision we created
(``revision_nonces``, kept by time so successor commits cannot evict it). Not
applied -> ``RevisionConflict`` (REST 409), nothing written. Neither a lock nor a
TTL is part of this guarantee.

The VECTOR (dense + BM25) is derived data (``embed(text)``). It may lag the text
for a moment but converges: after the text CAS the writer writes the vectors with
``update_vectors`` (which cannot create a missing point), re-reads the revision and,
if a later edit committed meanwhile, re-embeds THAT text and writes again (bounded
retries); only then it records ``vector_revision`` (filtered on the revision) and
clears ``vector_pending``. A crash or a stalled writer leaves ``vector_pending``
set; ``repair_stale_vectors`` (graph-worker cron) re-embeds such rows.

The per-memory Redis lock (``SET NX PX`` + token) is ONLY a contention reducer: it
avoids wasted embeds when two edits race. Correctness holds with Redis down, with
the lock expired, or with the lock lost mid-edit (tests cover each).

INVARIANTS of an edit (tests: ``tests/test_patch_revision.py``,
``tests/test_patch_revision_qdrant_local.py``, real stores in
``integration_tests/test_patch_revision_real_stores.py``):
* I1 - a conditional edit commits at exactly revision N as ONE state change
  (text + metadata + revision) or changes nothing, and says which.
* I2 - a committed edit is never reported as a conflict because of later writes.
* I3 - exactly one of N concurrent edits holding the same revision wins; no text is lost.
* I4 - the final vector is the embedding of the final text (convergence), and a
  delete during an edit is never resurrected.

DOCUMENTED LIMITS: (1) the dense/BM25 vector is eventually consistent with the
text (a reader can briefly see new text with the previous vector); a mismatch is
detectable (``vector_pending``) and repaired. (2) Writers that bypass
``patch_memory`` (dedup counters, the dreaming graph patcher) do not bump the
revision; they never change the patchable fields a revision guards.
"""

import logging
import uuid

from datetime import datetime, timezone
from config import settings
from index_format import distill_title
from savings_meter import stamp_tokens
from schemas import GLOBAL_CATEGORIES, MEMORY_CATEGORIES, MemoryVisibility, PROJECT_CATEGORIES, normalize_visibility
from memory.audit import _audit_log
from memory.groups import _build_group_id, _check_edit_permission

logger = logging.getLogger(__name__)


class RevisionConflict(Exception):
    """A PATCH carried an ``expected_revision`` that is no longer current.

    Nothing was written. ``current_revision`` is the row's revision at the
    time of the refusal, so a client can re-read, re-apply its change and retry.
    """

    def __init__(self, memory_id: str, expected: str, current: str):
        self.memory_id = memory_id
        self.expected_revision = expected
        self.current_revision = current
        super().__init__(
            f"Memory {memory_id} was modified since revision {expected} "
            f"(current revision is {current}); re-read it and retry"
        )


def revision_of(meta: dict | None) -> str:
    """The row's revision token: a per-memory write counter kept in ``metadata.revision``.

    Rows written before this existed carry no counter and read as ``"0"``. The
    counter is bumped by every ``patch_memory`` and ``retag_memories`` write, so
    a client holding a stale revision is refused. Guarantee boundary: a revision
    names ONE state among writers that take the per-memory edit lock
    (``patch_memory``, conditional or not, and ``retag_memories``); see the
    module docstring for the protocol, the lock's expiry behaviour and the
    writers that do not take it (dedup counters, the dreaming graph patcher).
    """
    try:
        return str(max(int((meta or {}).get("revision") or 0), 0))
    except (TypeError, ValueError):
        return "0"


def normalize_expected_revision(value) -> str | None:
    """Canonical form of a client-supplied revision (``None`` stays ``None``).

    Accepts the token as returned by reads, tolerating the quoting/``W/`` an
    HTTP ``ETag`` style echo would add. Anything else is a ``ValueError``.
    """
    if value is None:
        return None
    token = str(value).strip()
    if token.startswith("W/"):
        token = token[2:]
    token = token.strip('"')
    if not token.isascii() or not token.isdigit():
        raise ValueError("expected_revision must be a revision token as returned by a read of the memory")
    return str(int(token))


_LOCK_TTL_MS = 120_000      # contention reducer only (correctness does not depend on it)
_LOCK_WAIT_S = 5.0          # how long a writer waits for the lock before proceeding anyway
_LOCK_POLL_S = 0.05
_VECTOR_RETRIES = 3         # re-embed rounds when a later edit commits during our vector write
_NONCE_TTL_S = 600
_NONCE_CAP = 512

_RELEASE_LUA = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"


def _now_epoch() -> float:
    """Seconds since the epoch (one seam so tests can move the clock)."""
    return datetime.now(timezone.utc).timestamp()


def _redis_client():
    """Sync Redis client for the per-memory edit lock (one seam so tests inject an in-memory fake)."""
    import redis

    return redis.Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)


class _EditLock:
    """Best-effort ``SET NX PX`` lock with a token, released by compare-and-delete.

    A contention reducer: if Redis is down, or the lock cannot be taken within the
    wait budget, the edit PROCEEDS (the store-level compare-and-set decides).
    """

    def __init__(self, memory_id: str):
        self.key = f"ns:edit-lock:{memory_id}"
        self.memory_id = memory_id
        self.token = uuid.uuid4().hex
        self.client = None
        self.owned = False

    def __enter__(self):
        import time

        try:
            self.client = _redis_client()
            deadline = time.monotonic() + _LOCK_WAIT_S
            while not self.client.set(self.key, self.token, nx=True, px=_LOCK_TTL_MS):
                if time.monotonic() >= deadline:
                    logger.info("Edit lock for %s busy past the wait budget; proceeding (CAS decides)", self.memory_id)
                    return self
                time.sleep(_LOCK_POLL_S)
            self.owned = True
        except Exception as e:  # noqa: BLE001 - Redis down
            self.client = None
            logger.warning("Edit lock unavailable for %s; proceeding without it: %s", self.memory_id, e)
        return self

    def __exit__(self, *exc):
        if self.client is not None and self.owned:
            try:
                self.client.eval(_RELEASE_LUA, 1, self.key, self.token)
            except Exception as e:  # noqa: BLE001 - the TTL will release it
                logger.warning("Could not release the edit lock for %s: %s", self.memory_id, e)
        return False


def _cas_filter(memory_id: str, revision: str, parent_key: str = "metadata"):
    """Filter selecting exactly this point at exactly this revision (a missing point matches nothing)."""
    from qdrant_client.models import FieldCondition, Filter, HasIdCondition, IsEmptyCondition, MatchValue, PayloadField

    key = f"{parent_key}.revision"
    if int(revision) == 0:
        at_revision = Filter(should=[
            IsEmptyCondition(is_empty=PayloadField(key=key)),
            FieldCondition(key=key, match=MatchValue(value=0)),
        ])
    else:
        at_revision = Filter(must=[FieldCondition(key=key, match=MatchValue(value=int(revision)))])
    return Filter(must=[HasIdCondition(has_id=[memory_id]), at_revision])


def _unwrap_meta(payload) -> dict:
    meta = (payload or {}).get("metadata") or {}
    if isinstance(meta.get("metadata"), dict):
        meta = meta["metadata"]
    return meta


class EditMixin:
    """EditMixin for MemoryService (mechanical split — see memory_service.py)."""

    # ── Editing ─────────────────────────────────────────────
    #
    # Metadata keys a PATCH may touch inside payload["metadata"]. `scope` is
    # deliberately absent — it is always re-derived from category + project_id,
    # and `owner_user_id` is never editable.
    _PATCHABLE_META_KEYS = (
        "category", "project_id", "tags", "domain",
        "observation_type", "concepts", "confidence", "expires_at",
    )
    # Keys where an explicit null means "clear the field".
    _CLEARABLE_META_KEYS = frozenset(
        {"project_id", "tags", "domain", "observation_type", "concepts", "confidence", "expires_at"}
    )

    @staticmethod
    def _derive_scope(category: str | None, project_id: str | None) -> str:
        """Re-derive scope from the effective category + project_id (same rule as writes)."""
        if category in PROJECT_CATEGORIES:
            if not project_id:
                raise ValueError(
                    f"project_id is required for project category '{category}' "
                    "(set project_id in the same edit, or change the category)"
                )
            return "project"
        if category in GLOBAL_CATEGORIES:
            return "global"
        # flexible / adapter / null category: scope follows project_id
        return "project" if project_id else "global"

    def _apply_meta_changes(self, meta: dict, changes: dict) -> dict:
        """Merge PATCH changes into a copy of the nested metadata dict.

        Presence-keyed: only keys in ``changes`` are touched; a None value
        clears the key where legal. Category membership is checked at call
        time because adapters extend MEMORY_CATEGORIES at import.
        """
        new_meta = dict(meta)
        for key in self._PATCHABLE_META_KEYS:
            if key not in changes:
                continue
            value = changes[key]
            if value is None:
                if key not in self._CLEARABLE_META_KEYS:
                    raise ValueError(f"'{key}' cannot be cleared — provide a value or omit the field")
                new_meta.pop(key, None)
                continue
            if key == "category" and value not in MEMORY_CATEGORIES:
                raise ValueError(f"Invalid category: {value}")
            if key == "expires_at":
                value = value.isoformat() if hasattr(value, "isoformat") else str(value)
            new_meta[key] = value
        return new_meta

    @staticmethod
    def _text_fields(text: str, now_iso: str) -> dict:
        """The derived text fields mem0's own update writes (``hash``, BM25 ``text_lemmatized``)."""
        import hashlib

        from mem0.utils.lemmatization import lemmatize_for_bm25

        return {
            "data": text,
            "hash": hashlib.md5(text.encode()).hexdigest(),
            "text_lemmatized": lemmatize_for_bm25(text),
        }

    def _cas_set_payload(
        self, m, memory_id: str, fields: dict, new_meta: dict, old_meta: dict,
        current_revision: str, expected_revision: str | None, parent_key: str,
    ) -> None:
        """The compare-and-set: filtered ``set_payload``, applied only if the point exists at ``current_revision``.

        Applied-or-not is read back from the writer token recorded for the revision we create.
        """
        client = m.vector_store.client
        collection = settings.qdrant_collection
        token = uuid.uuid4().hex
        new_revision = int(current_revision) + 1
        now = _now_epoch()
        nonces = dict(old_meta.get("revision_nonces") or {})
        nonces[str(new_revision)] = {"n": token, "t": now}
        nonces = {r: e for r, e in nonces.items() if isinstance(e, dict) and e.get("t", 0) >= now - _NONCE_TTL_S}
        for old in sorted(nonces, key=int)[:-_NONCE_CAP]:
            del nonces[old]
        new_meta["revision"] = new_revision
        new_meta["revision_nonces"] = nonces
        for stale in ("revision_pending", "revision_pending_at", "revision_nonce"):
            new_meta.pop(stale, None)
        client.set_payload(
            collection_name=collection,
            payload=fields,
            points=_cas_filter(memory_id, current_revision, parent_key),
        )
        points = client.retrieve(
            collection_name=collection, ids=[memory_id], with_payload=True, with_vectors=False
        )
        if not points:
            raise LookupError(f"Memory {memory_id} not found")
        stored_payload = points[0].payload
        if not isinstance(stored_payload, dict):
            return  # a non-store double (unit-test mock): nothing to read back
        stored = _unwrap_meta(stored_payload)
        entry = (stored.get("revision_nonces") or {}).get(str(new_revision))
        if not isinstance(entry, dict) or entry.get("n") != token:
            raise RevisionConflict(memory_id, expected_revision or current_revision, revision_of(stored))

    @staticmethod
    def _record_edit_history(m, memory_id: str, old_payload: dict, text: str, now_iso: str) -> None:
        """The history row and entity-store upkeep mem0's own update does, after the text CAS applied."""
        try:
            m.db.add_history(
                memory_id,
                old_payload.get("data"),
                text,
                "UPDATE",
                created_at=old_payload.get("created_at"),
                updated_at=now_iso,
                actor_id=old_payload.get("actor_id"),
                role=old_payload.get("role"),
            )
            session_filters = {k: old_payload[k] for k in ("user_id", "agent_id", "run_id") if old_payload.get(k)}
            m._remove_memory_from_entity_store(memory_id, session_filters)
            m._link_entities_for_memory(memory_id, text, session_filters)
        except Exception as e:  # noqa: BLE001 - history/entity upkeep is best effort, as in mem0
            logger.warning("History/entity upkeep failed for %s: %s", memory_id, e)

    def _sync_vector(self, m, memory_id: str, text: str, embedding, text_revision: int, parent_key: str) -> bool:
        """Make the stored vector (dense + BM25) match the CURRENT text; returns True when converged.

        ``update_vectors`` cannot create a point, so a memory deleted meanwhile is never resurrected.
        After each write the revision is re-read: if a later edit committed, THAT text is
        embedded and written (bounded retries). The marker is cleared (and ``vector_revision``
        recorded) with a write filtered on the revision, so it never overwrites a newer state.
        """
        from qdrant_client.models import PointVectors

        client = m.vector_store.client
        collection = settings.qdrant_collection
        for _ in range(_VECTOR_RETRIES):
            vector = {"": embedding}
            if getattr(m.vector_store, "_has_bm25_slot", False):
                sparse = m.vector_store._encode_bm25(text)
                if sparse is not None:
                    vector["bm25"] = sparse
            try:
                client.update_vectors(
                    collection_name=collection, points=[PointVectors(id=memory_id, vector=vector)]
                )
            except Exception as e:  # noqa: BLE001 - missing point (deleted) or a transient store error
                if not client.retrieve(collection_name=collection, ids=[memory_id], with_payload=False, with_vectors=False):
                    logger.info("Memory %s was deleted during its edit; no vector written, nothing resurrected", memory_id)
                    return True
                logger.warning("Vector write for %s failed (left pending for repair): %s", memory_id, e)
                return False
            points = client.retrieve(collection_name=collection, ids=[memory_id], with_payload=True, with_vectors=False)
            if not points:
                return True                                  # deleted after our write: it is gone, not resurrected
            stored_payload = points[0].payload
            if not isinstance(stored_payload, dict):
                return True                                  # non-store double
            revision = int(revision_of(_unwrap_meta(stored_payload)))
            if revision == text_revision:
                client.set_payload(
                    collection_name=collection,
                    payload={"vector_revision": revision, "vector_pending": None, "vector_pending_at": None},
                    points=_cas_filter(memory_id, str(revision), parent_key),
                )
                return True
            # A later edit committed meanwhile: embed ITS text and write again.
            text = stored_payload.get("data", "")
            text_revision = revision
            embedding = m.embedding_model.embed(text, "update")
        return False                                         # still pending: repair_stale_vectors converges it

    def repair_stale_vectors(self, limit: int = 100, min_age_s: float = 60.0) -> dict:
        """Re-embed rows whose vector write did not finish (a crash or a stalled writer left ``vector_pending``)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        m = self._get_memory()
        client = m.vector_store.client
        collection = settings.qdrant_collection
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="vector_pending", match=MatchValue(value=True)),
                FieldCondition(key="vector_pending_at", range=Range(lt=_now_epoch() - min_age_s)),
            ]),
            limit=limit, with_payload=True, with_vectors=False,
        )
        repaired = failed = 0
        for pt in points:
            payload = pt.payload or {}
            text = payload.get("data", "")
            revision = int(revision_of(_unwrap_meta(payload)))
            parent_key = "metadata.metadata" if isinstance((payload.get("metadata") or {}).get("metadata"), dict) else "metadata"
            try:
                embedding = m.embedding_model.embed(text, "update")
                ok = self._sync_vector(m, str(pt.id), text, embedding, revision, parent_key)
            except Exception as e:  # noqa: BLE001
                logger.warning("Vector repair failed for %s: %s", pt.id, e)
                ok = False
            repaired += 1 if ok else 0
            failed += 0 if ok else 1
        return {"scanned": len(points), "repaired": repaired, "failed": failed}

    def patch_memory(
        self,
        memory_id: str,
        caller_user_id: str | None,
        changes: dict,
        expected_revision: str | None = None,
    ) -> dict:
        """Partially update a memory (see ``_patch_memory_locked`` for the full contract).

        A best-effort per-memory Redis lock only reduces contention; the write itself is a
        store-level compare-and-set (module docstring: guarantee, invariants, limits).
        """
        expected = normalize_expected_revision(expected_revision)
        with _EditLock(memory_id) as lock:
            return self._patch_memory_locked(memory_id, caller_user_id, changes, expected, lock)

    def _patch_memory_locked(
        self,
        memory_id: str,
        caller_user_id: str | None,
        changes: dict,
        expected_revision: str | None,
        lock: "_EditLock",
    ) -> dict:
        """Partially update a memory across the vector store and knowledge graph.

        ``changes`` is presence-keyed (built from the request's
        ``model_fields_set``): an explicit None clears the field where legal,
        an absent key is untouched. Returns::

            {"memory": MemoryResponse, "graph_job": dict | None,
             "graph": "unchanged" | "reingest_pending" | "reingest_incomplete"
                      | "migration_pending" | "migration_incomplete",
             "sibling_graph_jobs": list[dict],
             "sibling_graph_unrepaired_ids": list[str],
             "sibling_graph_repair_failed": bool}

        ``expected_revision`` (optional) is an optimistic-concurrency
        precondition: the revision token a read returned (``MemoryResponse.
        revision``). When given and no longer current, nothing is written and
        ``RevisionConflict`` is raised (REST 409). Absent keeps the historical
        last-writer-wins behaviour. Every successful patch bumps the revision.
        The write is ONE store-level compare-and-set carrying content, metadata and
        revision+1 (module docstring: guarantee, invariants, limits). Last-writer-wins
        for an unconditional edit means it reads the current revision and CASes on it;
        if another edit commits in between it answers 409 instead of overwriting.

        The caller is responsible for enqueuing ``graph_job`` on the graph
        queue — Graphiti work is minutes-slow and must never run inline on a
        request thread. Graph impact:

        - tags/category/v2 fields: not stored in Neo4j → vector-only patch.
        - project_id/visibility: part of the graph group_id partition →
          old edges are soft-expired here (fast, no LLM) and the content is
          re-ingested into the new group by the returned graph_job. Node
          group_ids are never mutated in place — entity nodes are shared
          across memories.
        - content: re-embedded here; the graph_job re-ingests so Graphiti's
          contradiction detection expires stale facts.

        ``"migration_incomplete"`` is a partition-migration variant of
        ``"migration_pending"``: the graph IS configured but the
        episode-precise cascade could not resolve/verify the OLD group's
        episode (so it may still be live there) — the caller still gets
        ``graph_job`` to re-ingest into the NEW group, but should also
        surface this distinctly (e.g. to an operator polling task status)
        rather than reporting a clean "migration_pending", which a pre-fix
        incident showed can silently hide a stuck old-group cascade.
        """
        m = self._get_memory()
        point = m.vector_store.get(vector_id=memory_id)
        if point is None:
            raise LookupError(f"Memory {memory_id} not found")
        payload = dict(getattr(point, "payload", None) or {})
        meta = payload.get("metadata") or {}
        # mem0 sometimes double-wraps metadata; unwrap before reading
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        meta = dict(meta)
        expected_revision = normalize_expected_revision(expected_revision)

        for key in ("content", "category", "visibility"):
            if key in changes and changes[key] is None:
                raise ValueError(f"'{key}' cannot be cleared — provide a value or omit the field")

        edits_content = "content" in changes and changes["content"] != payload.get("data", "")
        edits_visibility = "visibility" in changes
        _check_edit_permission(
            meta,
            payload.get("user_id", ""),
            caller_user_id,
            edits_content=edits_content,
            edits_visibility=edits_visibility,
        )

        # Precondition (after the permission gate so a caller who may not edit
        # never learns the row's revision from a refusal).
        current_revision = revision_of(meta)
        if expected_revision is not None and expected_revision != current_revision:
            raise RevisionConflict(memory_id, expected_revision, current_revision)

        # Passages are verbatim chunks of an ingested artifact — rewriting one
        # would silently diverge from the source document. Metadata edits are fine.
        if edits_content and meta.get("memory_kind") == "passage":
            raise ValueError(
                "Content edits are not allowed on 'passage' memories — they mirror "
                "an ingested artifact. Re-ingest the corrected source instead."
            )

        owner = meta.get("owner_user_id") or payload.get("user_id", "")
        old_visibility = meta.get("visibility") or MemoryVisibility.PRIVATE.value
        old_group = _build_group_id(old_visibility, owner, meta.get("project_id"))

        new_meta = self._apply_meta_changes(meta, changes)
        if edits_content or 'category' in changes:
            new_meta.pop('category_evidence', None)

        if edits_visibility:
            new_visibility = normalize_visibility(changes["visibility"])
            if new_visibility == MemoryVisibility.STANDARD.value:
                # Mirror store_raw's standard-tier gate: dictator-only, forced
                # global scope (standards are org-wide by definition).
                if not settings.standards_enabled:
                    raise PermissionError(
                        "The 'standard' visibility tier is disabled (set STANDARDS_ENABLED=true)."
                    )
                if not settings.is_dictator(caller_user_id):
                    raise PermissionError(
                        f"User {caller_user_id!r} is not authorized to write 'standard'-tier memories."
                    )
                new_meta.pop("project_id", None)
            new_meta["visibility"] = new_visibility
        else:
            new_visibility = normalize_visibility(old_visibility) or MemoryVisibility.PRIVATE.value

        new_meta["scope"] = self._derive_scope(new_meta.get("category"), new_meta.get("project_id"))
        # store_raw always writes the project_id key (None when global) — keep that shape.
        new_meta["project_id"] = new_meta.get("project_id")

        # Retrieval economics (C1): a content edit invalidates the write-time
        # title/token_estimate — refresh them from the new content.
        if edits_content:
            new_meta["title"] = distill_title(changes["content"])
            new_meta["token_estimate"] = stamp_tokens(changes["content"])

        now_iso = datetime.now(timezone.utc).isoformat()
        # Every write bumps the revision (even without a precondition) so a
        # concurrent holder of the old revision is refused.
        new_meta["revision"] = int(current_revision) + 1
        new_meta.pop("revision_pending", None)       # leftovers of the earlier lease protocol
        new_meta.pop("revision_pending_at", None)
        new_meta.pop("revision_nonce", None)
        new_meta.pop("revision_nonces", None)
        client = m.vector_store.client  # noqa: F841 - kept for symmetry with retag
        parent_key = "metadata.metadata" if isinstance((payload.get("metadata") or {}).get("metadata"), dict) else "metadata"
        embedding = None
        if edits_content:
            # The slow part (the embedder round trip) happens BEFORE the compare-and-set.
            embedding = m.embedding_model.embed(changes["content"], "update")
        # ONE store-level compare-and-set: text + metadata + revision+1 together, filtered on
        # id == memory AND metadata.revision == the revision we read. Not applied -> RevisionConflict.
        fields: dict = {"metadata": new_meta, "updated_at": now_iso}
        if edits_content:
            fields.update(self._text_fields(changes["content"], now_iso))
            fields["vector_pending"] = True                 # cleared once the vector matches this text
            fields["vector_pending_at"] = _now_epoch()
        self._cas_set_payload(
            m, memory_id, fields, new_meta, meta, current_revision, expected_revision, parent_key
        )
        if edits_content:
            self._record_edit_history(m, memory_id, payload, changes["content"], now_iso)
            self._sync_vector(m, memory_id, changes["content"], embedding, int(current_revision) + 1, parent_key)

        new_content = changes["content"] if edits_content else payload.get("data", "")
        new_group = _build_group_id(new_visibility, owner, new_meta.get("project_id"))

        graph_job = None
        graph_status = "unchanged"
        sibling_jobs: list[dict] = []
        sibling_unrepaired: list[str] = []
        sibling_repair_failed = False
        migrating = new_group != old_group
        if migrating or edits_content:
            # Partition migration: expire the memory's episode/edges in the
            # OLD group (episode-precise cascade first, substring fallback
            # only if no episode resolves — memory/provenance.py), then the
            # caller re-ingests into the new group via the graph queue.
            #
            # A pre-fix incident here was a SILENT no-op: the old substring
            # routine could fail to match anything, log one line, and this
            # method would still report a clean "migration_pending" — the
            # caller (and any operator polling task status) had no signal
            # that the old group's content was still live. Surface that
            # distinguishably instead: "migration_incomplete" whenever the
            # graph is actually configured but the cascade could not
            # resolve (and therefore could not verify) the episode.
            #
            # A content-only edit takes the same cascade: the OLD text lives
            # verbatim in the memory's episode, which episode recall serves
            # (on by default), so it is expired exactly as for a migration
            # (most-restrictive) before the new content is re-ingested.
            graph_available = bool(self._graphiti and self._bridge)
            cascade_result: dict = {"resolved": False}
            try:
                # The moved row may share a conversation episode with sibling
                # facts. The whole episode is expired (its body holds the moved
                # fact verbatim) — the siblings are repaired: stale ref cleared
                # and a single-fact re-ingest job returned for the caller to
                # enqueue. The moved row is re-ingested via ``graph_job`` below.
                batch = self._cascade_rows_once(
                    [{
                        "id": memory_id,
                        "mem": {"memory": payload.get("data", ""), "metadata": meta, "user_id": owner},
                        "group_id": old_group,
                    }],
                    skip_job_ids={memory_id},
                )
                cascade_result = batch["results"].get(memory_id) or cascade_result
                sibling_jobs = batch["graph_jobs"]
                sibling_unrepaired = batch["sibling_unrepaired_ids"]
                sibling_repair_failed = batch["sibling_repair_failed"]
            except Exception as e:
                logger.warning(f"Graph edge expiration failed for {memory_id} (non-critical): {e}")
            unverified = graph_available and not cascade_result.get("resolved")
            if unverified:
                logger.warning(
                    "%s for memory=%s could not resolve/verify "
                    "graph cleanup in old group=%r — old content may still be "
                    "live in the graph until a follow-up cleanup runs",
                    "Partition migration" if migrating else "Content edit",
                    memory_id, old_group,
                )
            if migrating:
                graph_status = "migration_incomplete" if unverified else "migration_pending"
            else:
                graph_status = "reingest_incomplete" if unverified else "reingest_pending"
        if graph_status != "unchanged":
            graph_job = {
                "memory_id": memory_id,
                "content": new_content,
                "user_id": owner,
                "project_id": new_meta.get("project_id"),
                "visibility": new_visibility,
                "source_ref": meta.get("source_ref"),
            }

        _audit_log.info(
            "memory_patched",
            memory_id=memory_id,
            caller=caller_user_id,
            fields=sorted(changes.keys()),
            graph=graph_status,
        )
        return {
            "memory": self.get_memory(memory_id),
            "graph_job": graph_job,
            "graph": graph_status,
            # Additive: re-ingest jobs for sibling facts that shared the moved
            # row's (now expired) conversation episode. Empty for single-fact rows.
            "sibling_graph_jobs": sibling_jobs,
            "sibling_graph_unrepaired_ids": sibling_unrepaired,
            "sibling_graph_repair_failed": sibling_repair_failed,
        }

    def retag_memories(
        self,
        caller_user_id: str | None,
        filters: dict,
        ops: dict,
        dry_run: bool = False,
    ) -> dict:
        """Bulk-apply metadata operations to memories matching a filter set.

        ``filters``: scope / category / project_id / visibility / tags_contains
        (AND semantics). ``ops`` is presence-keyed: add_tags, remove_tags,
        set_category, set_project_id (explicit None clears the project),
        set_workspace (explicit None clears it; WT6 migration tool).

        Visibility and content are deliberately NOT bulk-editable. Other
        users' private memories never enter the candidate set (the scroll
        filter restricts to shared/standard pools + the caller's own rows),
        so counts can't leak their existence. Per-row permission and
        category-matrix violations are skipped and counted, not fatal.

        Returns ``{matched, updated, skipped_forbidden, skipped_invalid,
        graph_jobs, dry_run}`` — the caller enqueues ``graph_jobs`` (produced
        when a project change moves a memory between graph groups).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if ops.get("set_category") and ops["set_category"] not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category: {ops['set_category']}")

        m = self._get_memory()
        client = m.vector_store.client
        collection = settings.qdrant_collection

        must = []
        if filters.get("scope"):
            must.append(FieldCondition(key="metadata.scope", match=MatchValue(value=filters["scope"])))
        if filters.get("category"):
            must.append(FieldCondition(key="metadata.category", match=MatchValue(value=filters["category"])))
        if filters.get("project_id"):
            must.append(FieldCondition(key="metadata.project_id", match=MatchValue(value=filters["project_id"])))
        if filters.get("visibility"):
            must.append(FieldCondition(key="metadata.visibility", match=MatchValue(value=filters["visibility"])))
        for tag in filters.get("tags_contains") or []:
            must.append(FieldCondition(key="metadata.tags", match=MatchValue(value=tag)))
        # Service-side backstop for the request-boundary sweep guard: worker and
        # MCP paths hand this raw dicts, and falsey filter values ("" / []) are
        # skipped above — without this check they'd select every candidate row.
        if not must:
            raise ValueError(
                "At least one effective filter is required — refusing an unfiltered retag sweep"
            )
        # Candidate set = shared OR standard OR the caller's own rows. This keeps
        # other users' PRIVATE memories out entirely (no permission-skip count
        # leakage; legacy null-visibility rows are covered by the user_id clause).
        should = [
            FieldCondition(key="metadata.visibility", match=MatchValue(value=MemoryVisibility.SHARED.value)),
            FieldCondition(key="metadata.visibility", match=MatchValue(value=MemoryVisibility.STANDARD.value)),
            Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=caller_user_id or ""))]),
        ]
        scroll_filter = Filter(must=must, should=should)

        add_tags = list(ops.get("add_tags") or [])
        remove_tags = set(ops.get("remove_tags") or [])
        now_iso = datetime.now(timezone.utc).isoformat()

        matched = updated = skipped_forbidden = skipped_invalid = 0
        skipped_busy_ids: list[str] = []
        graph_jobs: list[dict] = []
        # Zero-effect detection (audit): a memory whose episode can't be
        # resolved must be COUNTED and its id REPORTED, never silently
        # absorbed by a per-row try/except that only logs.
        graph_migrations_unresolved_ids: list[str] = []
        migrations: list[dict] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                matched += 1
                payload = pt.payload or {}
                meta = payload.get("metadata") or {}
                if isinstance(meta.get("metadata"), dict):
                    meta = meta["metadata"]
                meta = dict(meta)
                try:
                    _check_edit_permission(
                        meta, payload.get("user_id", ""), caller_user_id,
                        edits_content=False, edits_visibility=False,
                    )
                except PermissionError:
                    skipped_forbidden += 1
                    continue

                new_meta = dict(meta)
                if add_tags or remove_tags:
                    tags = [t for t in (meta.get("tags") or []) if t not in remove_tags]
                    tags.extend(t for t in add_tags if t not in tags)
                    if tags:
                        new_meta["tags"] = tags
                    else:
                        new_meta.pop("tags", None)
                if ops.get("set_category"):
                    new_meta["category"] = ops["set_category"]
                    new_meta.pop('category_evidence', None)
                if "set_project_id" in ops:
                    if ops["set_project_id"] is None:
                        new_meta.pop("project_id", None)
                    else:
                        new_meta["project_id"] = ops["set_project_id"]
                # WT6: workspace housekeeping operation
                if "set_workspace" in ops:
                    if ops["set_workspace"] is None:
                        new_meta.pop("workspace", None)
                    else:
                        new_meta["workspace"] = ops["set_workspace"]

                changed = (
                    (new_meta.get("tags") or None) != (meta.get("tags") or None)
                    or new_meta.get("category") != meta.get("category")
                    or new_meta.get("project_id") != meta.get("project_id")
                    or new_meta.get("workspace") != meta.get("workspace")
                )
                if not changed:
                    continue  # matched, nothing to change
                try:
                    new_meta["scope"] = self._derive_scope(
                        new_meta.get("category"), new_meta.get("project_id")
                    )
                except ValueError:
                    skipped_invalid += 1
                    continue
                new_meta["project_id"] = new_meta.get("project_id")

                # Standard-tier invariant: only a dictator may touch standard
                # memories or create new ones (changing the category of an
                # existing standard is a standard-affecting edit).
                current_vis = meta.get("visibility") or MemoryVisibility.PRIVATE.value
                if current_vis == MemoryVisibility.STANDARD.value and not settings.is_dictator(caller_user_id):
                    skipped_forbidden += 1
                    continue

                updated += 1
                if dry_run:
                    continue
                # A retag changes patchable fields: it goes through the SAME compare-and-set as a PATCH,
                # filtered on the revision the scroll saw, so a row that moved on (or was deleted) since
                # is skipped and reported (``skipped_busy_ids``) rather than overwritten from a stale scroll.
                busy = False
                parent_key = "metadata.metadata" if isinstance((payload.get("metadata") or {}).get("metadata"), dict) else "metadata"
                try:
                    with _EditLock(str(pt.id)):
                        self._cas_set_payload(
                            m, str(pt.id),
                            {"metadata": new_meta, "updated_at": now_iso}, new_meta, meta,
                            revision_of(meta), None, parent_key,
                        )
                except (RevisionConflict, LookupError):
                    busy = True        # the row moved on (or is gone) since the scroll: not retagged, reported
                if busy:
                    updated -= 1
                    skipped_busy_ids.append(str(pt.id))
                    continue
                owner = meta.get("owner_user_id") or payload.get("user_id", "")
                visibility = meta.get("visibility") or MemoryVisibility.PRIVATE.value
                # WT6: include workspace in group_id derivation
                old_group = _build_group_id(visibility, owner, meta.get("project_id"), meta.get("workspace"))
                new_group = _build_group_id(visibility, owner, new_meta.get("project_id"), new_meta.get("workspace"))
                if new_group != old_group:
                    # Cascaded after the scroll, once per distinct episode
                    # (rows of one conversation share an episode), with
                    # sibling repair that skips every migrated row (each
                    # gets its own re-ingest job below).
                    migrations.append({
                        "id": str(pt.id),
                        "mem": {"memory": payload.get("data", ""), "metadata": meta, "user_id": owner},
                        "group_id": old_group,
                    })
                    graph_jobs.append({
                        "memory_id": str(pt.id),
                        "content": payload.get("data", ""),
                        "user_id": owner,
                        "project_id": new_meta.get("project_id"),
                        "visibility": visibility,
                        "source_ref": meta.get("source_ref"),
                    })
            if offset is None:
                break

        sibling_unrepaired: list[str] = []
        sibling_repair_failed = False
        sibling_jobs_count = 0
        if migrations:
            graph_available = bool(self._graphiti and self._bridge)
            batch = {"results": {}, "graph_jobs": [], "sibling_unrepaired_ids": []}
            try:
                batch = self._cascade_rows_once(
                    migrations, skip_job_ids={r["id"] for r in migrations}
                )
            except Exception as e:
                logger.warning(f"Retag graph cascade failed (non-critical): {e}")
            for row in migrations:
                res = batch["results"].get(row["id"]) or {"resolved": False}
                if graph_available and not res.get("resolved"):
                    logger.warning(
                        "Retag partition migration for memory=%s could not "
                        "resolve/verify graph cleanup in old group=%r — old "
                        "content may still be live in the shared graph pool",
                        row["id"], row["group_id"],
                    )
                    graph_migrations_unresolved_ids.append(row["id"])
            # Sibling re-ingests ride the same list the callers already enqueue.
            graph_jobs.extend(batch["graph_jobs"])
            sibling_jobs_count = len(batch["graph_jobs"])
            sibling_unrepaired = list(batch["sibling_unrepaired_ids"])
            sibling_repair_failed = bool(batch.get("sibling_repair_failed"))

        _audit_log.info(
            "memories_retagged",
            caller=caller_user_id,
            filters={k: v for k, v in filters.items() if v is not None},
            ops={k: v for k, v in ops.items()},
            matched=matched,
            updated=updated,
            skipped_forbidden=skipped_forbidden,
            skipped_invalid=skipped_invalid,
            graph_migrations=len(graph_jobs),
            graph_migrations_unresolved=len(graph_migrations_unresolved_ids),
            dry_run=dry_run,
        )
        return {
            "matched": matched,
            "updated": updated,
            "skipped_forbidden": skipped_forbidden,
            "skipped_invalid": skipped_invalid,
            # Rows another edit held (lock busy / lost) or that changed since the scroll: not retagged.
            "skipped_busy_ids": skipped_busy_ids,
            "graph_jobs": graph_jobs,
            # Rows whose old-group episode could not be resolved/verified —
            # the substring fallback ran but its effect is unverifiable.
            # Never silently dropped; a caller/operator can act on these ids.
            "graph_migrations_unresolved_ids": graph_migrations_unresolved_ids,
            # Of graph_jobs, how many re-ingest SIBLING facts of a cascaded
            # conversation episode (the rest re-ingest the moved rows).
            "sibling_graph_jobs_count": sibling_jobs_count,
            "sibling_graph_unrepaired_ids": sibling_unrepaired,
            "sibling_graph_repair_failed": sibling_repair_failed,
            "dry_run": dry_run,
        }
