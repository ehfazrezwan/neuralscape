"""Write-path throughput & integrity fixes — audit 27, Cluster 4 (items 20-24).

Failing-first regression tests for:

- **#20 (gateway embeds)** — ``OpenAIEmbedding.embed_batch`` with
  ``embedding_batch_size=1`` (the gateway's single-input-only Vertex endpoint)
  fans per-item calls out onto a bounded thread pool instead of N serial HTTP
  round trips; per-item retry before failing; failures name the input index.
- **#21 (storage idempotency)** — ``_batch_store_facts`` content-hash dedups
  before insert (an ARQ re-run inserts zero new points, bumps times_derived on
  the survivors); graph episodes carry a deterministic content-keyed
  idempotency name and ``graph.add`` is skipped when the episode already exists.
- **#22 (windowed extraction)** — long conversations are split into
  ~30-message windows (2-message overlap), one extraction call per window;
  short conversations stay byte-identical to the unwindowed path; one failed
  window degrades to a partial result instead of zeroing the session.
- **#24 (summarizers off the fast queue)** — ``process_session_summary`` is
  registered on the graph worker and enqueued onto the graph queue.
- **#20 (checkpoints)** — ``store_raw_batch`` two-pass: dedup first, then ONE
  ``embed_batch`` call for all new items, per-item inserts, order preserved.
"""

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.openai import OpenAIEmbedding

from config import settings
from memory_service import MemoryService, content_hash
from schemas import MemoryResponse


# ──────────────────────────────────────────────
# Shared fakes
# ──────────────────────────────────────────────


def _vec(text: str) -> list[float]:
    """Deterministic per-text fake embedding."""
    return [float(sum(text.encode()) % 97)] * 8


class _FakeEmbeddingsAPI:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        return self._outer._create(**kwargs)


class _FakeOpenAIClient:
    """Thread-safe fake OpenAI client with optional per-call latency/failures.

    MagicMock call recording is not thread-safe; the parallel embed path
    exercises real threads, so counting must be locked.
    """

    def __init__(self, latency: float = 0.0, failures: dict[str, int] | None = None):
        self.latency = latency
        self.failures = dict(failures or {})  # text -> remaining failures
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()
        self.embeddings = _FakeEmbeddingsAPI(self)

    def _create(self, **kwargs):
        inputs = list(kwargs["input"])
        with self._lock:
            self.calls.append(inputs)
            for t in inputs:
                if self.failures.get(t, 0) > 0:
                    self.failures[t] -= 1
                    raise Exception(f"Error code: 503 - transient for {t!r}")
        if self.latency:
            time.sleep(self.latency)
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=_vec(t)) for i, t in enumerate(inputs)]
        )


def _embedder(client: _FakeOpenAIClient, **config_kwargs) -> OpenAIEmbedding:
    config = BaseEmbedderConfig(
        model="google-vertex/gemini-embedding-001",
        api_key="test-key",
        openai_base_url="http://localhost:9/v1",
        embedding_dims=768,
        **config_kwargs,
    )
    emb = OpenAIEmbedding(config)
    emb.client = client
    return emb


@pytest.fixture(autouse=True)
def _fast_embed_retry(monkeypatch):
    """Zero the per-item retry backoff so retry tests don't sleep."""
    import mem0.embeddings.openai as openai_mod

    if hasattr(openai_mod, "_EMBED_RETRY_BACKOFF_S"):
        monkeypatch.setattr(openai_mod, "_EMBED_RETRY_BACKOFF_S", 0.0)


# ──────────────────────────────────────────────
# #20 — parallel per-item gateway embeds
# ──────────────────────────────────────────────


class TestParallelGatewayEmbeds:
    def test_sixteen_texts_complete_in_two_waves(self):
        """16 single-input embeds through an 8-worker pool ≈ 2 waves of wall
        time — NOT 16 serial round trips."""
        latency = 0.15
        texts = [f"fact number {i}" for i in range(16)]
        emb = _embedder(_FakeOpenAIClient(latency=latency), embedding_batch_size=1)

        start = time.monotonic()
        result = emb.embed_batch(texts)
        elapsed = time.monotonic() - start

        assert result == [_vec(t) for t in texts]  # order preserved
        assert len(emb.client.calls) == 16
        assert all(len(c) == 1 for c in emb.client.calls)
        # Serial would be 16 * 0.15 = 2.4s; two 8-wide waves ≈ 0.3s.
        assert elapsed < 1.2, f"per-item embeds still serial ({elapsed:.2f}s)"

    def test_single_text_stays_direct(self):
        emb = _embedder(_FakeOpenAIClient(), embedding_batch_size=1)
        assert emb.embed_batch(["only one"]) == [_vec("only one")]
        assert len(emb.client.calls) == 1

    def test_per_item_transient_failure_retried(self):
        texts = [f"t{i}" for i in range(4)]
        emb = _embedder(
            _FakeOpenAIClient(failures={"t2": 1}), embedding_batch_size=1
        )
        result = emb.embed_batch(texts)
        assert result == [_vec(t) for t in texts]
        assert len(emb.client.calls) == 5  # 4 + 1 retry

    def test_persistent_failure_names_the_input_index(self):
        texts = [f"t{i}" for i in range(4)]
        emb = _embedder(
            _FakeOpenAIClient(failures={"t2": 99}), embedding_batch_size=1
        )
        with pytest.raises(Exception, match=r"index 2"):
            emb.embed_batch(texts)

    def test_single_input_rejection_fallback_is_parallel(self):
        """The batched-call-rejected fallback routes through the same parallel
        path (it is exactly the gateway situation, just undetected upfront)."""
        latency = 0.15
        texts = [f"fb {i}" for i in range(8)]

        class _RejectBatches(_FakeOpenAIClient):
            def _create(self, **kwargs):
                if len(kwargs["input"]) > 1:
                    with self._lock:
                        self.calls.append(list(kwargs["input"]))
                    raise Exception(
                        "Error code: 400 - accepts only one input per request"
                    )
                return super()._create(**kwargs)

        emb = _embedder(_RejectBatches(latency=latency))  # default batching
        start = time.monotonic()
        result = emb.embed_batch(texts)
        elapsed = time.monotonic() - start
        assert result == [_vec(t) for t in texts]
        # 1 rejected batch + 8 per-item calls, one 8-wide wave ≈ 0.15s
        assert len(emb.client.calls) == 9
        assert elapsed < 0.9, f"fallback still serial ({elapsed:.2f}s)"


# ──────────────────────────────────────────────
# Service fixture (mirrors tests/test_memory_service.py)
# ──────────────────────────────────────────────


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.embedding_model.embed_batch.side_effect = (
        lambda texts, **kw: [[0.1] * 768 for _ in texts]
    )
    # The attach hook parks 10s on the mocked bridge's dead future per stored
    # memory (pre-existing test-suite quirk) — not under test here, stub it.
    svc._attach_memory_id_to_graph_nodes = MagicMock(name="attach")
    return svc


def _mock_extraction(svc, payloads: list) -> MagicMock:
    """Mock the Gemini client; ``payloads`` is one entry per expected call —
    either a list of fact strings or an Exception to raise."""
    client = MagicMock()
    svc._genai_model = client

    responses = []
    for p in payloads:
        if isinstance(p, Exception):
            responses.append(p)
        else:
            responses.append(MagicMock(text=json.dumps({"facts": p})))
    if len(responses) == 1:
        client.models.generate_content.return_value = responses[0]
    else:
        client.models.generate_content.side_effect = responses
    return client


def _stateful_hash_dedup(service):
    """Wire _find_by_content_hash + insert into an in-memory hash store so
    dedup behaves like a real Qdrant collection across calls."""
    store: dict[tuple, MemoryResponse] = {}

    def fake_find(user_id, content_hash, scope, project_id=None, visibility=None):
        return store.get((user_id, content_hash, scope, project_id))

    def record_insert(vectors, ids, payloads):
        for mid, p in zip(ids, payloads):
            meta = p["metadata"]
            store[(p["user_id"], p["hash"], meta["scope"], meta["project_id"])] = (
                MemoryResponse(
                    id=mid,
                    memory=p["data"],
                    category=meta["category"],
                    scope=meta["scope"],
                    project_id=meta["project_id"],
                )
            )

    service._find_by_content_hash = fake_find
    service._memory.vector_store.insert.side_effect = record_insert
    service._bump_times_derived = MagicMock(name="bump")
    service._revive_if_tombstoned = MagicMock(name="revive", return_value=False)
    return store


# ──────────────────────────────────────────────
# #21 — _batch_store_facts content-hash dedup
# ──────────────────────────────────────────────


FACTS = [
    ("preference", "Prefers dark mode in every editor"),
    ("technical_skill", "Expert in Python 3.12 and FastAPI"),
    ("personal_fact", "Based in Dhaka, works UTC+6 hours"),
]


class TestBatchStoreFactsDedup:
    def test_second_pass_inserts_zero_new_points(self, service):
        _stateful_hash_dedup(service)

        first = service._batch_store_facts(facts=FACTS, user_id="ehfaz")
        assert service._memory.vector_store.insert.call_count == 1

        second = service._batch_store_facts(facts=FACTS, user_id="ehfaz")
        # No new insert, no new embed call — everything dedup'd.
        assert service._memory.vector_store.insert.call_count == 1
        assert service._memory.embedding_model.embed_batch.call_count == 1
        assert {r.id for r in second} == {r.id for r in first}
        assert len(second) == len(FACTS)

    def test_dedup_hits_bump_times_derived_and_revive(self, service):
        _stateful_hash_dedup(service)
        service._batch_store_facts(facts=FACTS, user_id="ehfaz")
        service._batch_store_facts(facts=FACTS, user_id="ehfaz")
        assert service._bump_times_derived.call_count == len(FACTS)
        assert service._revive_if_tombstoned.call_count == len(FACTS)

    def test_in_batch_duplicates_collapse_to_one_insert(self, service):
        """Window overlap re-extracts the same fact — one row, not two."""
        _stateful_hash_dedup(service)
        facts = [FACTS[0], FACTS[0]]
        results = service._batch_store_facts(facts=facts, user_id="ehfaz")
        insert_kwargs = service._memory.vector_store.insert.call_args.kwargs
        assert len(insert_kwargs["payloads"]) == 1
        assert len(results) == 1

    def test_different_users_do_not_dedup(self, service):
        _stateful_hash_dedup(service)
        service._batch_store_facts(facts=FACTS, user_id="ehfaz")
        service._batch_store_facts(facts=FACTS, user_id="robb")
        assert service._memory.vector_store.insert.call_count == 2
        assert service._bump_times_derived.call_count == 0


# ──────────────────────────────────────────────
# #21 — graph episode idempotency key
# ──────────────────────────────────────────────


MESSAGES = [
    {"role": "user", "content": "I prefer dark mode everywhere"},
    {"role": "assistant", "content": "Noted — dark mode it is."},
]


def _wire_episode_tracking(service):
    """graph.add records episode names; the existence probe consults them —
    an in-memory stand-in for the Cypher lookup."""
    added: set[str] = set()

    def fake_exists(group_id, episode_name):
        return episode_name in added

    def record_add(*args, **kwargs):
        added.add(kwargs.get("episode_name"))
        return {"deleted_entities": [], "added_entities": []}

    service._graph_episode_exists = fake_exists
    service._memory.graph.add.side_effect = record_add
    return added


class TestGraphEpisodeIdempotency:
    def test_second_run_skips_graph_add(self, service):
        _mock_extraction(service, [["[preference] Prefers dark mode"]] * 2)
        added = _wire_episode_tracking(service)

        service.extract_and_store(messages=MESSAGES, user_id="ehfaz")
        service.extract_and_store(messages=MESSAGES, user_id="ehfaz")

        assert service._memory.graph.add.call_count == 1
        assert len(added) == 1

    def test_episode_name_is_deterministic_and_content_keyed(self, service):
        _mock_extraction(service, [["[preference] Prefers dark mode"]])
        _wire_episode_tracking(service)
        service.extract_and_store(messages=MESSAGES, user_id="ehfaz")
        name = service._memory.graph.add.call_args.kwargs["episode_name"]
        assert name.startswith("mem0_episode_")
        # Deterministic: not the timestamp form (no ISO separator chars).
        assert ":" not in name and "T" not in name

    def test_different_conversations_get_different_keys(self, service):
        _mock_extraction(service, [["[preference] A fact here"]] * 2)
        added = _wire_episode_tracking(service)
        other = [{"role": "user", "content": "Completely different conversation"}]
        service.extract_and_store(messages=MESSAGES, user_id="ehfaz")
        service.extract_and_store(messages=other, user_id="ehfaz")
        assert service._memory.graph.add.call_count == 2
        assert len(added) == 2

    def test_exists_probe_failure_fails_open(self, service):
        """A broken lookup degrades to today's behavior (episode added).

        The service fixture's bridge/graphiti are MagicMocks, so the REAL
        ``_graph_episode_exists`` Cypher probe blows up internally — it must
        swallow that and return False so the graph write still happens.
        """
        _mock_extraction(service, [["[preference] Prefers dark mode"]])
        service.extract_and_store(messages=MESSAGES, user_id="ehfaz")
        assert service._memory.graph.add.call_count == 1

    def test_memorygraph_add_forwards_episode_name(self):
        from mem0.memory.graphiti_memory import MemoryGraph

        mg = MemoryGraph.__new__(MemoryGraph)
        mg._update_communities = False
        mg._indices_built = True  # skip _ensure_indices
        mg.graphiti = MagicMock()
        mg.graphiti.add_episode = AsyncMock(
            return_value=SimpleNamespace(edges=[], nodes=[])
        )
        mg._bridge = SimpleNamespace(run=lambda coro: asyncio.run(coro))

        mg.add(data="hello", filters={"group_id": "g1"}, episode_name="mem0_episode_deadbeef")
        assert mg.graphiti.add_episode.await_args.kwargs["name"] == "mem0_episode_deadbeef"

        mg.add(data="hello", filters={"group_id": "g1"})
        assert mg.graphiti.add_episode.await_args.kwargs["name"].startswith("mem0_episode_")


# ──────────────────────────────────────────────
# #22 — windowed conversation extraction
# ──────────────────────────────────────────────


def _msgs(n: int) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _no_operator_guidance(monkeypatch):
    """Hermetic: operator guidance resolution must not touch Redis/env."""
    import extraction_settings

    monkeypatch.setattr(
        extraction_settings, "resolve_instructions", lambda *a, **k: None
    )


class TestWindowedExtraction:
    def test_window_constants(self):
        assert settings.extraction_window_messages == 30
        assert settings.extraction_window_overlap == 2

    def test_split_boundaries(self):
        from prompts import split_into_windows

        msgs = _msgs(100)
        windows = split_into_windows(msgs, 30, 2)
        assert len(windows) == 4
        assert windows[0] == msgs[0:30]
        assert windows[1] == msgs[28:58]
        assert windows[2] == msgs[56:86]
        assert windows[3] == msgs[84:100]
        # Short conversations: exactly the input, untouched.
        short = _msgs(30)
        assert split_into_windows(short, 30, 2) == [short]

    def test_short_conversation_is_byte_identical_single_call(self, service):
        from prompts import build_extraction_messages

        msgs = _msgs(20)
        client = _mock_extraction(service, [["[preference] Prefers dark mode"]])
        _wire_episode_tracking(service)

        service.extract_and_store(messages=msgs, user_id="ehfaz")

        assert client.models.generate_content.call_count == 1
        sent = client.models.generate_content.call_args.kwargs["contents"]
        assert sent == build_extraction_messages(msgs)[0]["content"]

    def test_long_conversation_fans_out_and_unions_facts(self, service):
        msgs = _msgs(100)
        per_window_facts = [
            [f"[preference] Window fact {i} about topic {i}"] for i in range(4)
        ]
        client = _mock_extraction(service, per_window_facts)
        _wire_episode_tracking(service)

        stored = service.extract_and_store(messages=msgs, user_id="ehfaz")

        assert client.models.generate_content.call_count >= 4
        assert len(stored) == 4  # facts from every window unioned
        assert {m.memory for m in stored} == {
            f"Window fact {i} about topic {i}" for i in range(4)
        }
        # Still exactly ONE graph episode for the whole conversation.
        assert service._memory.graph.add.call_count == 1

    def test_one_failed_window_keeps_the_rest_and_reports_partial(self, service):
        msgs = _msgs(100)
        payloads = [
            ["[preference] Window fact 0 about topic 0"],
            Exception("boom mid-window"),
            ["[preference] Window fact 2 about topic 2"],
            ["[preference] Window fact 3 about topic 3"],
        ]
        _mock_extraction(service, payloads)
        _wire_episode_tracking(service)

        stored, stats = service.extract_and_store(
            messages=msgs, user_id="ehfaz", return_stats=True
        )
        assert len(stored) == 3
        assert stats["windows_total"] == 4
        assert stats["windows_failed"] == 1

    def test_all_windows_failing_still_raises(self, service):
        msgs = _msgs(100)
        _mock_extraction(service, [Exception("API down")] * 4)
        with pytest.raises(Exception, match="API down"):
            service.extract_and_store(messages=msgs, user_id="ehfaz")
        service._memory.vector_store.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_surfaces_partial_extraction(self):
        import worker

        mem = MemoryResponse(id="m1", memory="x", category="preference")
        svc = MagicMock(name="MemoryService")
        svc.extract_and_store.return_value = (
            [mem],
            {"windows_total": 4, "windows_failed": 1, "window_errors": ["window 2/4: boom"]},
        )
        ctx = {"service": svc, "redis": MagicMock(enqueue_job=AsyncMock())}
        result = await worker.process_memory_store(
            ctx, [{"role": "user", "content": "hi"}], "u1"
        )
        assert result["partial_extraction"] is True
        assert result["windows_failed"] == 1
        assert result["windows_total"] == 4

    @pytest.mark.asyncio
    async def test_worker_result_clean_when_all_windows_succeed(self):
        import worker

        mem = MemoryResponse(id="m1", memory="x", category="preference")
        svc = MagicMock(name="MemoryService")
        svc.extract_and_store.return_value = (
            [mem],
            {"windows_total": 1, "windows_failed": 0, "window_errors": []},
        )
        ctx = {"service": svc, "redis": MagicMock(enqueue_job=AsyncMock())}
        result = await worker.process_memory_store(
            ctx, [{"role": "user", "content": "hi"}], "u1"
        )
        assert "partial_extraction" not in result


# ──────────────────────────────────────────────
# #24 — session summarizers off the fast queue
# ──────────────────────────────────────────────


class TestSessionSummaryQueue:
    @pytest.mark.asyncio
    async def test_enqueue_targets_graph_queue(self, monkeypatch):
        import session_summarizer as ss
        import worker

        monkeypatch.setattr(ss, "record_messages", lambda *a, **k: (20, ["short"]))
        ctx = {"redis": MagicMock(enqueue_job=AsyncMock())}
        await worker._note_session_messages(
            ctx, "u1", "sess-1", [{"role": "user", "content": "m"}]
        )
        kwargs = ctx["redis"].enqueue_job.await_args.kwargs
        assert kwargs["_queue_name"] == settings.graph_queue_name

    def test_registered_on_graph_worker_not_fast(self):
        import worker

        assert worker.process_session_summary in worker.GraphWorkerSettings.functions
        assert worker.process_session_summary not in worker.WorkerSettings.functions


# ──────────────────────────────────────────────
# #20 — store_raw_batch two-pass (checkpoints)
# ──────────────────────────────────────────────


class TestStoreRawBatchTwoPass:
    def _items(self, contents):
        return [
            {"content": c, "user_id": "ehfaz", "category": "preference"}
            for c in contents
        ]

    def test_one_embed_batch_call_dedup_first(self, service):
        contents = [f"Checkpoint fact number {i} about topic {i}" for i in range(10)]
        dup_rows = {
            content_hash(contents[i]): MemoryResponse(
                id=f"existing-{i}", memory=contents[i],
                category="preference", scope="global",
            )
            for i in (2, 7)
        }
        service._find_by_content_hash = (
            lambda user_id, content_hash, scope, project_id=None, visibility=None:
            dup_rows.get(content_hash)
        )
        service._bump_times_derived = MagicMock()
        service._revive_if_tombstoned = MagicMock(return_value=False)

        embed_calls: list[list[str]] = []

        def fake_embed_batch(texts, memory_action="add"):
            embed_calls.append(list(texts))
            return [[0.1] * 768 for _ in texts]

        service._memory.embedding_model.embed_batch.side_effect = fake_embed_batch

        results = service.store_raw_batch(self._items(contents))

        # ONE embed_batch call, only the 8 new texts.
        assert len(embed_calls) == 1
        assert len(embed_calls[0]) == 8
        service._memory.embedding_model.embed.assert_not_called()
        # 8 per-item inserts, 2 dedup bumps.
        assert service._memory.vector_store.insert.call_count == 8
        assert service._bump_times_derived.call_count == 2
        # Per-item results preserved, in input order.
        assert [r.memory for r in results] == contents
        assert results[2].id == "existing-2"
        assert results[7].id == "existing-7"

    def test_all_duplicates_skips_embedding_entirely(self, service):
        contents = ["Same old fact stored before"]
        existing = MemoryResponse(
            id="existing-0", memory=contents[0], category="preference", scope="global"
        )
        service._find_by_content_hash = lambda **kw: existing
        service._bump_times_derived = MagicMock()
        service._revive_if_tombstoned = MagicMock(return_value=False)

        results = service.store_raw_batch(self._items(contents))
        service._memory.embedding_model.embed_batch.assert_not_called()
        service._memory.vector_store.insert.assert_not_called()
        assert [r.id for r in results] == ["existing-0"]

    def test_in_batch_duplicate_items_collapse(self, service):
        contents = ["Repeated checkpoint fact", "Repeated checkpoint fact"]
        service._find_by_content_hash = lambda **kw: None
        service._bump_times_derived = MagicMock()
        service._revive_if_tombstoned = MagicMock(return_value=False)

        results = service.store_raw_batch(self._items(contents))
        assert service._memory.vector_store.insert.call_count == 1
        assert len(results) == 2  # both items answered, same surviving row
        assert results[0].id == results[1].id
        assert service._bump_times_derived.call_count == 1

    def test_bad_item_skipped_without_blocking(self, service):
        service._find_by_content_hash = lambda **kw: None
        items = self._items(["Good fact one here", "Good fact two here"])
        items.insert(1, {"content": "Bad", "user_id": "e", "category": "NOPE"})
        results = service.store_raw_batch(items)
        assert len(results) == 2
        assert service._memory.vector_store.insert.call_count == 2
