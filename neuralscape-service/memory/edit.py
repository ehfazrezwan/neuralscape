"""Edit path: patch_memory and bulk retag housekeeping.

Mechanically extracted from memory_service.py (mixins-with-facade split);
the revision precondition and the store-level compare-and-set were added on top.

GUARANTEE (what correctness rests on): the text and its revision change in ONE call. The commit is a
single TOP-LEVEL ``set_payload`` carrying ``data``/``hash``/``text_lemmatized``/``updated_at`` together with
``ns_revision`` (= revision+1; the revision lives beside the text it versions), the writer ``ns_token``,
the ``ns_nonces`` entry and the metadata delta parked in ``ns_meta_delta`` (``ns_meta_pending``), filtered on
``id == <memory> AND ns_revision == <expected>`` (a legacy row without ``ns_revision`` matches through its
``metadata.revision`` mirror, or the absence of both for revision 0). Qdrant applies it only to a point that
exists and still matches: a stale writer cannot overwrite a later edit, a deleted point cannot be re-created,
and a top-level merge cannot erase nested metadata written by other writers (``graph_episode_uuid``,
``times_derived``, tombstones, ...). Whether it applied is read back from our nonce entry (kept by time, so
successor commits cannot evict it). Not applied -> ``RevisionConflict`` (REST 409) and NOTHING was written.
Neither a lock nor a TTL is part of this guarantee, and a reader never sees a revision that does not match
the text it sees (``revision_of(meta, top)`` reads ``ns_revision`` first).
The DERIVED metadata follows in two filter-scoped calls (filter: ``ns_revision == new AND ns_token == ours``):
a nested ``set_payload(key="metadata")`` of only the changed keys (a removed key is written as ``null``; the
engine does not clear nested keys through ``delete_payload``) plus the ``metadata.revision`` mirror, then
clearing ``ns_meta_pending``. If the process dies between the calls the parked delta stays pending:
``repair_stale_vectors`` finishes it, and any later edit folds a pending delta into its own base, so it is
never lost; until then the nested metadata lags the text (like the vector).

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

DOCUMENTED LIMITS: (1) the dense/BM25 vector and the nested metadata are eventually consistent with the
text (a reader can briefly see new text with the previous vector, or with the previous nested metadata; an
EXCEPTION in the deferred apply, not only a crash, leaves the nested metadata behind the text until repair, and
the edit still succeeds). What is NOT deferred: the access-control keys (``visibility``, ``project_id``,
``scope``, ``category``) of an edit that does not widen the audience are written BEFORE the text commit, so the
new text is never readable by the old audience; an edit that widens visibility defers them. If such a commit
then loses the race, the narrowing stays (the fail-closed direction). A mismatch is detectable (``vector_pending``, ``ns_meta_pending``) and repaired. A crash inside the few
milliseconds between a stale vector write and its post-check is not detectable. (2) Writers that bypass
``patch_memory`` (dedup counters, the dreaming graph patcher) do not bump the revision; they never change the
patchable fields a revision guards. (3) The repair scans filter on fields that are not payload-indexed (the
index entries are waiting for the owner), so each scan reads the collection; rows PROCESSED per run are bounded
(500) and a failing row stops being retried after 5 attempts. (4) The search paths that build responses from
raw points (``memory/search.py``) report the ``metadata.revision`` mirror, which can only LAG the true
revision (the safe direction: a conditional writer holding it is refused).
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


def revision_of(meta: dict | None, top=None) -> str:
    """The row's revision token (a per-memory write counter).

    ``top`` is the TOP-LEVEL payload field ``ns_revision``, which lives next to the text it versions
    (``data``/``hash``) and is written in the same single call; it wins when present. Rows that
    predate it fall back to the ``metadata.revision`` mirror, and rows with neither read as ``"0"``.
    Every read path that reports a revision passes ``top`` when it has the top-level payload, so the
    revision shown always matches the text shown (module docstring has the protocol and the limits).
    """
    for value in (top, (meta or {}).get("revision")):
        if value is None:
            continue
        try:
            return str(max(int(value), 0))
        except (TypeError, ValueError):
            continue
    return "0"


def _text_revision(payload: dict) -> int:
    """The revision at which the TEXT last changed (``ns_text_revision``); a metadata-only edit does not move it.

    Rows whose text predates this field (never edited through this path) fall back to the row revision.
    """
    try:
        return int(payload.get("ns_text_revision"))
    except (TypeError, ValueError):
        return int(_payload_revision(payload))


def _text_filter(memory_id: str, text_revision: int):
    """Filter selecting this point while its text is still the one written at ``text_revision``."""
    from qdrant_client.models import FieldCondition, Filter, HasIdCondition, MatchValue

    return Filter(must=[
        HasIdCondition(has_id=[memory_id]),
        FieldCondition(key="ns_text_revision", match=MatchValue(value=int(text_revision))),
    ])


def _payload_revision(payload) -> str:
    """Revision of a raw Qdrant payload (top-level ``ns_revision``, else the ``metadata.revision`` mirror)."""
    return revision_of(_unwrap_meta(payload), (payload or {}).get("ns_revision"))


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


_REDIS = None


def _redis_client():
    """Sync Redis client for the per-memory edit lock (one shared client/pool; one seam so tests inject a fake)."""
    global _REDIS
    import redis

    if _REDIS is None:
        _REDIS = redis.Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)
    return _REDIS


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


def _cas_filter(memory_id: str, revision: str, parent_key: str = "metadata", token: str | None = None):
    """Filter selecting exactly this point at exactly this revision (a missing point matches nothing).

    The revision lives at the TOP level (``ns_revision``); a legacy row that has none matches through
    its ``metadata.revision`` mirror (or the absence of both, for revision 0). With ``token`` it also
    requires ``ns_token == token``: the follow-up calls of an edit apply only to the state THIS
    writer's first call created.
    """
    from qdrant_client.models import FieldCondition, Filter, HasIdCondition, IsEmptyCondition, MatchValue, PayloadField

    rev = int(revision)
    legacy_mirror = (
        Filter(should=[
            IsEmptyCondition(is_empty=PayloadField(key=f"{parent_key}.revision")),
            FieldCondition(key=f"{parent_key}.revision", match=MatchValue(value=0)),
        ])
        if rev == 0
        else Filter(must=[FieldCondition(key=f"{parent_key}.revision", match=MatchValue(value=rev))])
    )
    at_revision = Filter(should=[
        FieldCondition(key="ns_revision", match=MatchValue(value=rev)),
        Filter(must=[IsEmptyCondition(is_empty=PayloadField(key="ns_revision")), legacy_mirror]),
    ])
    must = [HasIdCondition(has_id=[memory_id]), at_revision]
    if token is not None:
        must.append(FieldCondition(key="ns_token", match=MatchValue(value=token)))
    return Filter(must=must)


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

    @staticmethod
    def _effective_meta(payload: dict) -> tuple[dict, dict]:
        """(stored nested metadata, effective metadata = stored + a pending, not-yet-applied delta)."""
        stored = dict(_unwrap_meta(payload))
        eff = dict(stored)
        if payload.get("ns_meta_pending") and isinstance(payload.get("ns_meta_delta"), dict):
            for k, v in payload["ns_meta_delta"].items():
                if v is None:
                    eff.pop(k, None)
                else:
                    eff[k] = v
        return stored, eff

    def _cas_set_payload(
        self, m, memory_id: str, text_fields: dict | None, new_meta: dict, stored_meta: dict, payload: dict,
        current_revision: str, expected_revision: str | None, parent_key: str,
    ) -> None:
        """The compare-and-set: ONE top-level filter-scoped ``set_payload`` carries text AND revision.

        Call A (the commit): ``data``/``hash``/``text_lemmatized``/``updated_at``, ``ns_revision`` = revision+1,
        the writer ``ns_token``, the ``ns_nonces`` entry, and the metadata delta parked in ``ns_meta_delta`` with
        ``ns_meta_pending`` set, filtered on ``id == memory AND revision == current`` (the revision is the TOP-LEVEL
        ``ns_revision``; legacy rows match through their ``metadata.revision`` mirror). Top-level ``set_payload``
        merges keys, so it can never erase nested metadata written by other writers. Applied-or-not is read back
        from our nonce entry (kept by time, so successors cannot evict it); not applied -> ``RevisionConflict``
        and nothing was written.
        Then the DERIVED metadata is applied, filtered on ``ns_revision == new AND ns_token == ours``:
        B writes only the changed keys (nested ``set_payload(key="metadata")``, removed keys as ``null``, plus the
        ``metadata.revision`` mirror) and C clears the pending marker. A crash between the calls leaves
        ``ns_meta_pending`` + the parked delta, which ``repair_stale_vectors`` finishes, and any later edit folds the
        pending delta into its own base, so it is never lost. A reader never sees a revision that does not match its text.
        """
        client = m.vector_store.client
        collection = settings.qdrant_collection
        token = uuid.uuid4().hex
        new_revision = int(current_revision) + 1
        now = _now_epoch()
        nonces = dict(payload.get("ns_nonces") or {})
        nonces[str(new_revision)] = {"n": token, "t": now}
        nonces = {r: e for r, e in nonces.items() if isinstance(e, dict) and e.get("t", 0) >= now - _NONCE_TTL_S}
        for old in sorted(nonces, key=int)[:-_NONCE_CAP]:
            del nonces[old]
        new_meta["revision"] = new_revision
        for stale in ("revision_pending", "revision_pending_at", "revision_nonce", "revision_nonces", "edit_token"):
            new_meta.pop(stale, None)
        delta = {k: v for k, v in new_meta.items() if stored_meta.get(k) != v}
        delta.update({k: None for k in stored_meta if k not in new_meta})       # removed keys -> null
        fields = dict(text_fields or {})
        if "data" in fields:                                  # a content edit: the TEXT revision moves
            fields["ns_text_revision"] = new_revision
            fields["vector_repair_attempts"] = None           # a new edit resets the repair retry budget
        fields.update({
            "ns_revision": new_revision, "ns_token": token, "ns_nonces": nonces,
            "ns_meta_pending": True, "ns_meta_pending_at": now, "ns_meta_delta": delta,
            "ns_meta_repair_attempts": None,
        })
        client.set_payload(
            collection_name=collection, payload=fields,
            points=_cas_filter(memory_id, current_revision, parent_key),
        )
        points = client.retrieve(collection_name=collection, ids=[memory_id], with_payload=True, with_vectors=False)
        if not points:
            raise LookupError(f"Memory {memory_id} not found")
        stored_payload = points[0].payload
        if isinstance(stored_payload, dict):             # (a non-store double in unit tests has nothing to read back)
            entry = (stored_payload.get("ns_nonces") or {}).get(str(new_revision))
            if not isinstance(entry, dict) or entry.get("n") != token:
                raise RevisionConflict(memory_id, expected_revision or current_revision, _payload_revision(stored_payload))
        try:
            self._apply_pending_meta(client, collection, memory_id, new_revision, token, delta, parent_key)
        except Exception as e:  # noqa: BLE001
            # The edit is COMMITTED (text + revision). A store error in the deferred metadata apply must not turn the
            # response into an error or skip the history/graph work: ``ns_meta_pending`` + the parked delta stay for
            # ``repair_stale_vectors`` (and any later edit folds them in).
            logger.warning("Deferred metadata apply failed for %s (left pending for repair): %s", memory_id, e)

    _ACCESS_KEYS = ("visibility", "project_id", "scope", "category")
    _VISIBILITY_RANK = {"private": 0, "shared": 1, "standard": 2}

    def _narrow_access_before_text(self, m, memory_id, new_meta, stored_meta, current_revision, parent_key) -> dict:
        """Move the access-control keys BEFORE the commit when an edit does not widen the audience; returns what was written.

        The text commit and the nested metadata apply are separate calls, and every pool filter reads
        ``metadata.visibility`` / ``project_id`` / ``scope``: a narrowing edit (shared -> private, or a project move)
        must not expose its NEW text to the OLD audience for the instant (or, on a failed apply, the hours until
        repair) between them. So those keys are written first, conditional on the same revision. An edit that WIDENS
        visibility keeps them deferred (the old text reaching the wider audience first is what the owner is
        sharing anyway). Applies to metadata-only edits too: their deferred apply can fail as well, and a failed
        narrowing must not report success while the row stays readable. A project/scope move with the SAME
        visibility is written first as well; that is safe: it does not widen who may read the memory (visibility is
        unchanged), the destination pool's readers see the memory they were meant to see, and what leaves is the
        old pool's access. Known edge: if the commit then loses the race the narrowing stays applied (the
        fail-closed direction; it is logged).
        """
        delta = {
            k: new_meta.get(k) for k in self._ACCESS_KEYS if new_meta.get(k) != stored_meta.get(k)
        }
        if not delta:
            return {}
        rank = self._VISIBILITY_RANK
        old_rank = rank.get(str(stored_meta.get("visibility") or "private"), 0)
        new_rank = rank.get(str(new_meta.get("visibility") or "private"), 0)
        if new_rank > old_rank:
            return {}
        m.vector_store.client.set_payload(
            collection_name=settings.qdrant_collection,
            payload=delta,
            points=_cas_filter(memory_id, current_revision, parent_key),
            key=parent_key,
        )
        return delta

    @staticmethod
    def _apply_pending_meta(client, collection, memory_id, revision, token, delta, parent_key) -> None:
        """B + C: apply the parked metadata delta to the row state (revision, token) and clear the marker."""
        flt = _cas_filter(memory_id, str(revision), parent_key, token=token)
        if delta:
            client.set_payload(collection_name=collection, payload=delta, points=flt, key=parent_key)
        client.set_payload(
            collection_name=collection,
            payload={"ns_meta_pending": None, "ns_meta_pending_at": None, "ns_meta_delta": None},
            points=flt,
        )

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

    @staticmethod
    def _bm25_sparse(m, text: str):
        """The BM25 sparse vector exactly as mem0 builds it (from the lemmatized text); ``None`` = unavailable."""
        from mem0.utils.lemmatization import lemmatize_for_bm25

        return m.vector_store._encode_bm25(lemmatize_for_bm25(text) or text)

    def _sync_vector(self, m, memory_id: str, text: str, embedding, text_revision: int, parent_key: str) -> bool:
        """Make the stored vector (dense + BM25) match the CURRENT text; returns True when converged.

        ``text_revision`` is the revision at which ``text`` was written (``ns_text_revision``): a
        metadata-only edit does not move it, so it never triggers a needless re-embed. ``update_vectors``
        cannot create a point, so a memory deleted meanwhile is never resurrected. Before each write the
        text revision is checked: if a later CONTENT edit already committed (and may have cleared the
        marker), THAT text is embedded and the marker re-asserted first. After each write it is re-read
        once more; the marker is cleared (and ``vector_revision`` recorded) by a write filtered on the
        text revision, so it never overwrites a newer state. Returns False (marker left set) when the BM25
        encoder is unavailable or retries ran out.
        """
        from qdrant_client.models import PointVectors

        client = m.vector_store.client
        collection = settings.qdrant_collection
        for _ in range(_VECTOR_RETRIES):
            points = client.retrieve(collection_name=collection, ids=[memory_id], with_payload=True, with_vectors=False)
            if not points:
                return True                                  # deleted: nothing to write, nothing resurrected
            stored_payload = points[0].payload
            if isinstance(stored_payload, dict):
                current = _text_revision(stored_payload)
                if current != text_revision:                 # a later content edit committed: adopt ITS text
                    text, text_revision = stored_payload.get("data", ""), current
                    embedding = m.embedding_model.embed(text, "update")
                    client.set_payload(                      # its writer may have cleared the marker already
                        collection_name=collection,
                        payload={"vector_pending": True, "vector_pending_at": _now_epoch()},
                        points=_text_filter(memory_id, current),
                    )
            vector = {"": embedding}
            if getattr(m.vector_store, "_has_bm25_slot", False):
                sparse = self._bm25_sparse(m, text)
                if sparse is None:
                    logger.warning("BM25 encoder unavailable for %s; vector left pending for repair", memory_id)
                    return False
                vector["bm25"] = sparse
            try:
                client.update_vectors(collection_name=collection, points=[PointVectors(id=memory_id, vector=vector)])
            except Exception as e:  # noqa: BLE001 - missing point (deleted) or a transient store error
                if not client.retrieve(collection_name=collection, ids=[memory_id], with_payload=False, with_vectors=False):
                    logger.info("Memory %s was deleted during its edit; no vector written, nothing resurrected", memory_id)
                    return True
                logger.warning("Vector write for %s failed (left pending for repair): %s", memory_id, e)
                return False
            points = client.retrieve(collection_name=collection, ids=[memory_id], with_payload=True, with_vectors=False)
            if not points:
                return True
            stored_payload = points[0].payload
            if not isinstance(stored_payload, dict):
                return True                                  # non-store double
            if _text_revision(stored_payload) == text_revision:
                client.set_payload(
                    collection_name=collection,
                    payload={"vector_revision": text_revision, "vector_pending": None, "vector_pending_at": None,
                             "vector_repair_attempts": None},
                    points=_text_filter(memory_id, text_revision),
                )
                return True
            # A later content edit committed during our write: loop re-reads, adopts its text, re-asserts the marker.
        return False                                         # still pending: repair_stale_vectors converges it

    def repair_stale_vectors(self, limit: int = 500, min_age_s: float = 60.0, max_attempts: int = 5) -> dict:
        """Converge what an edit left unfinished. Bounded per run, paginated, with a per-row retry budget.

        COST (stated honestly): ``vector_pending`` / ``vector_revision`` / ``ns_*`` are not payload-indexed (the
        index entries wait for the owner), so each filtered scroll below scans the collection; the number of ROWS it
        processes per run is bounded by ``limit`` (500) and each failing row stops being retried after
        ``max_attempts``. On a healthy store every scan is empty and nothing is embedded.

        0. rows whose derived metadata was parked (``ns_meta_pending``) and not applied (a crash between an edit's
           calls): finish them from the parked delta (idempotent, filtered on the row's own revision + token);
        1. rows still ``vector_pending`` (a crash or a stalled writer) older than ``min_age_s``;
        2. rows whose ``vector_revision`` differs from their TEXT revision (``ns_text_revision``; a metadata-only edit
           does not move it) that the marker missed, scanned from a rotating cursor so healthy rows do not eat the
           budget every run.
        Both vector passes honour ``max_attempts``; a row that exhausts it is logged.
        """
        from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition, MatchValue, PayloadField, Range

        m = self._get_memory()
        client = m.vector_store.client
        collection = settings.qdrant_collection
        attempts_ok = Filter(should=[
            IsEmptyCondition(is_empty=PayloadField(key="vector_repair_attempts")),
            FieldCondition(key="vector_repair_attempts", range=Range(lt=max_attempts)),
        ])
        pending_filter = Filter(must=[
            FieldCondition(key="vector_pending", match=MatchValue(value=True)),
            FieldCondition(key="vector_pending_at", range=Range(lt=_now_epoch() - min_age_s)),
            attempts_ok,
        ])
        mismatch_filter = Filter(
            must=[attempts_ok], must_not=[IsEmptyCondition(is_empty=PayloadField(key="vector_revision"))]
        )
        meta_filter = Filter(must=[
            FieldCondition(key="ns_meta_pending", match=MatchValue(value=True)),
            FieldCondition(key="ns_meta_pending_at", range=Range(lt=_now_epoch() - min_age_s)),
        ])
        meta_repaired = 0
        meta_filter_budgeted = Filter(must=[
            meta_filter,
            Filter(should=[
                IsEmptyCondition(is_empty=PayloadField(key="ns_meta_repair_attempts")),
                FieldCondition(key="ns_meta_repair_attempts", range=Range(lt=max_attempts)),
            ]),
        ])
        offset = None
        budget = limit
        while budget > 0:
            points, offset = client.scroll(
                collection_name=collection, scroll_filter=meta_filter_budgeted, limit=min(100, budget),
                offset=offset, with_payload=True, with_vectors=False,
            )
            for pt in points:
                budget -= 1
                payload = pt.payload or {}
                parent = "metadata.metadata" if isinstance((payload.get("metadata") or {}).get("metadata"), dict) else "metadata"
                revision, token = int(payload.get("ns_revision") or 0), payload.get("ns_token")
                try:
                    self._apply_pending_meta(
                        client, collection, str(pt.id), revision, token, payload.get("ns_meta_delta") or {}, parent,
                    )
                    meta_repaired += 1
                except Exception as e:  # noqa: BLE001
                    attempts = int(payload.get("ns_meta_repair_attempts") or 0) + 1
                    logger.warning("Metadata repair failed for %s (attempt %d/%d): %s", pt.id, attempts, max_attempts, e)
                    if attempts >= max_attempts:
                        logger.error("Metadata repair attempts exhausted for %s; it stays pending for an operator", pt.id)
                    try:
                        client.set_payload(
                            collection_name=collection, payload={"ns_meta_repair_attempts": attempts},
                            points=_cas_filter(str(pt.id), str(revision), parent, token=token),
                        )
                    except Exception:  # noqa: BLE001
                        pass
            if offset is None or not points:
                break

        scanned = repaired = failed = 0
        seen: set[str] = set()
        cursor = getattr(self, "_repair_cursor", None)
        for flt, only_mismatch in ((pending_filter, False), (mismatch_filter, True)):
            offset = cursor if only_mismatch else None
            budget = limit
            wrapped = False
            while budget > 0:
                points, next_offset = client.scroll(
                    collection_name=collection, scroll_filter=flt, limit=min(100, budget),
                    offset=offset, with_payload=True, with_vectors=False,
                )
                for pt in points:
                    budget -= 1
                    payload = pt.payload or {}
                    if str(pt.id) in seen:
                        continue
                    text_rev = _text_revision(payload)
                    if only_mismatch and payload.get("vector_revision") == text_rev and not payload.get("vector_pending"):
                        continue                                  # healthy: its vector matches its text
                    seen.add(str(pt.id))
                    scanned += 1
                    parent_key = "metadata.metadata" if isinstance((payload.get("metadata") or {}).get("metadata"), dict) else "metadata"
                    try:
                        text = payload.get("data", "")
                        embedding = m.embedding_model.embed(text, "update")
                        ok = self._sync_vector(m, str(pt.id), text, embedding, text_rev, parent_key)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Vector repair failed for %s: %s", pt.id, e)
                        ok = False
                    if ok:
                        repaired += 1
                        continue
                    failed += 1
                    attempts = int(payload.get("vector_repair_attempts") or 0) + 1
                    if attempts >= max_attempts:
                        logger.error("Vector repair attempts exhausted for %s (%d); it stays pending for an operator", pt.id, attempts)
                    try:
                        client.set_payload(
                            collection_name=collection,
                            payload={"vector_repair_attempts": attempts, "vector_pending": True,
                                     "vector_pending_at": _now_epoch()},
                            points=_text_filter(str(pt.id), text_rev),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                offset = next_offset
                if offset is None or not points:
                    wrapped = True
                    break
            if only_mismatch:
                self._repair_cursor = None if wrapped else offset      # resume where this run stopped
        return {"scanned": scanned, "repaired": repaired, "failed": failed, "meta_repaired": meta_repaired}

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
        # ``stored_meta`` = what is on disk; ``meta`` = stored + a crash-pending metadata delta from an
        # earlier edit (so a later edit never loses it).
        stored_meta, meta = self._effective_meta(payload)
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
        current_revision = revision_of(meta, payload.get("ns_revision"))
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
        # Store-level compare-and-set (see ``_cas_set_payload``): metadata delta + revision first, then
        # the text fields, both filter-scoped. Not applied -> RevisionConflict, nothing written.
        text_fields: dict = {"updated_at": now_iso}
        # Access-control keys that do not widen the audience are written FIRST for every edit (content or
        # metadata-only): a deferred apply that fails must never leave the row readable by the old audience.
        narrowed = self._narrow_access_before_text(
            m, memory_id, new_meta, stored_meta, current_revision, parent_key
        )
        if edits_content:
            text_fields.update(self._text_fields(changes["content"], now_iso))
            text_fields["vector_pending"] = True            # cleared once the vector matches this text
            text_fields["vector_pending_at"] = _now_epoch()
        try:
            self._cas_set_payload(
                m, memory_id, text_fields, new_meta, stored_meta, payload, current_revision, expected_revision, parent_key
            )
        except RevisionConflict:
            if narrowed:
                logger.warning(
                    "Edit of %s lost the race AFTER its access-control keys were narrowed (fail-closed): the row "
                    "keeps the narrower access %s although its edit was rejected", memory_id, narrowed,
                )
            raise
        if edits_content:
            self._record_edit_history(m, memory_id, payload, changes["content"], now_iso)
            # The edit is COMMITTED: a failure in the vector sync must never turn the response into
            # an error or skip the graph work below. It leaves ``vector_pending`` for repair.
            try:
                self._sync_vector(m, memory_id, changes["content"], embedding, int(current_revision) + 1, parent_key)
            except Exception as e:  # noqa: BLE001
                logger.warning("Vector sync after a committed edit failed for %s (left pending for repair): %s", memory_id, e)

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
                stored_meta, meta = self._effective_meta(payload)
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
                            {"updated_at": now_iso}, new_meta, stored_meta, payload,
                            revision_of(meta, payload.get("ns_revision")), None, parent_key,
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
