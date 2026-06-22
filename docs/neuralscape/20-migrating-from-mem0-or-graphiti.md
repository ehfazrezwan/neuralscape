# Migrating to Neuralscape from mem0 or Graphiti

*Practical backfill guide for existing users. Neuralscape (NS) is an opinionated, already-assembled memory system built **on** mem0 and Graphiti — so if you already run either, your data has a home here. Verified against the NS ingestion API, mem0 2.0.2, and graphiti 0.28.2.*

> **TL;DR**
> - **mem0 user?** Export your memories with `Memory.get_all`, then POST them to NS's `POST /v1/memories/raw/batch`. Content-hash dedup makes re-runs idempotent.
> - **Graphiti user?** Two paths: **remap `group_id`s in place** (fast, lossless, graph-only) using the shipped `scripts/migrate_graph_groups.py`, or **re-ingest episodes** through `POST /v1/memories` (gets you NS's vector + categorization too).
> - Pick based on whether you want *graph-only continuity* (remap) or the *full NS experience* (re-ingest).

---

## NS ingestion surface (what you import into)

All write endpoints are async: they return **202 + `task_id`**, processed by the ARQ worker; poll `GET /v1/memories/status/{task_id}`. (They fall back to **200 + memories** if Redis is down.)

| Endpoint | LLM? | Use for | Handler |
|---|---|---|---|
| `POST /v1/memories` | yes (Gemini extracts + categorizes) | raw conversation/episode text you want NS to structure | `extract_and_store` (`memory_service.py:454`) |
| `POST /v1/memories/raw` | no | one already-clean, pre-categorized fact | `store_raw` (`memory_service.py:571`) |
| `POST /v1/memories/raw/batch` | no | bulk import (≤50 items/call) | `store_raw_batch` (`memory_service.py:892`) |

Key `raw` fields (`schemas.py:262`): `content` (required, 1–10000 chars), `category` (**required**, one of the 13 in `MEMORY_CATEGORIES`), `user_id`, `scope` (default `"global"`; `"project"` requires `project_id`), `visibility` (defaults per category — personal categories → private), `source_type` (use **`"imported"`** — it's a valid vocab value), plus optional `tags`, `domain`, `observation_type`, `concepts`, `confidence`, `expires_at`.

Two facts that shape every migration:
- **`created_at` cannot be overridden** — `store_raw` always stamps "now" (`memory_service.py:643`). Original timestamps survive only if you fold them into `content` or `tags`.
- **Dedup is on `md5(content) + user_id + scope`** (`_find_by_content_hash`, `memory_service.py:644`). Identical re-runs are idempotent; mutating content (e.g. embedding a volatile timestamp) defeats that.

---

## Path 1 — mem0 → Neuralscape

### What you can export
OSS mem0 has **no** dump helper, but `Memory.get_all(filters={"user_id": uid}, top_k=BIG)` (`main.py:1016`) returns each memory as `{id, memory (text), hash, created_at, updated_at, user_id/agent_id/run_id, metadata}`. Note `get_all` does **not** paginate (single Qdrant scroll, `qdrant.py:543`) — use a large `top_k` or scroll Qdrant directly with offsets for big stores. **OSS mem0 stores no categories**, so you supply one on import.

### Field mapping

| mem0 | NS (`/v1/memories/raw`) | Mapping |
|---|---|---|
| `memory` / `data` | `content` | direct (skip if empty / >10k) |
| *(none)* | `category` | **you choose** — default `personal_fact` (faithful, PRIVATE), or classify client-side |
| `user_id` | `user_id` | direct → becomes NS owner + private namespace |
| *(none)* | `scope` | `global` recommended unless bucketing by project |
| `agent_id`/`run_id` | `agent_id`/`run_id` | provenance |
| `created_at` | `tags` | **lossy** — store as `mem0_created:<ts>` tag; not the real timestamp |
| custom `metadata` | `tags` | flatten string values (max 20) |
| import marker | `source_type="imported"` | makes imports queryable/distinguishable |

### Recommended recipe — raw batch (faithful, cheap, idempotent)
Prefer `/v1/memories/raw/batch` over the LLM path: it stores text verbatim 1:1, whereas routing through `/v1/memories` lets Gemini rephrase/split/merge/junk-filter (`_is_junk_fact`, `memory_service.py:502`) — better categorization, lower fidelity. Only use the LLM path if you specifically want NS to re-structure messy text.

```python
import time, itertools, requests
from mem0 import Memory

NS, HEADERS = "http://localhost:8199", {"Authorization": "Bearer <NS_TOKEN>"}
DEFAULT_CATEGORY = "personal_fact"   # must be in MEMORY_CATEGORIES

def to_ns(rec):
    text = (rec.get("memory") or "").strip()
    if not text or len(text) > 10_000: return None
    tags = []
    if rec.get("created_at"): tags.append(f"mem0_created:{rec['created_at']}")
    for k, v in (rec.get("metadata") or {}).items():
        if isinstance(v, str) and len(tags) < 20: tags.append(f"{k}:{v}")
    return {"content": text, "category": DEFAULT_CATEGORY, "scope": "global",
            "user_id": rec.get("user_id"), "source_type": "imported", "tags": tags or None}

def chunks(it, n):
    it = iter(it)
    while (b := list(itertools.islice(it, n))): yield b

def backfill(user_id):
    rows = Memory().get_all(filters={"user_id": user_id}, top_k=10_000)["results"]
    items = [x for r in rows if (x := to_ns(r))]
    for batch in chunks(items, 50):                      # batch cap = 50
        r = requests.post(f"{NS}/v1/memories/raw/batch",
                          json={"memories": batch}, headers=HEADERS); r.raise_for_status()
        tid = r.json().get("task_id")
        if tid:                                          # poll the async task
            for _ in range(120):
                s = requests.get(f"{NS}/v1/memories/status/{tid}", headers=HEADERS).json()
                if s.get("status") in ("completed","failed","ok"): break
                time.sleep(1)
        time.sleep(0.25)                                 # throttle worker + embeddings

backfill("alice")
```

### Lossy / caveats
- Original `created_at`/`updated_at`, mem0 `id`, and `hash` are not preserved (NS mints its own).
- No real categories from OSS mem0 — all imports get your default unless you classify client-side first.
- **mem0's graph relationships are not transferred** — NS rebuilds its own Graphiti graph from each ingested `content`.
- Visibility follows the category default (`schemas.py:194`); `personal_fact` → private (safe). A project/episodic category would silently become `shared`.

---

## Path 2 — Graphiti → Neuralscape

### NS's group_id scheme (the thing that matters)
NS gates every graph read/write by `group_id` (`_build_group_id`, `memory_service.py:206-241`):

| visibility | project | group_id |
|---|---|---|
| private | — | `user--{user_id}` |
| private | set | `user--{user_id}--project--{project_id}` |
| shared | — | `shared` |
| shared | set | `shared--project--{project_id}` |

Reads union `["user--{uid}", "shared", …]` (`_get_group_ids`, `:244-259`). **Data under any other group_id is invisible to NS.** Your existing Graphiti data almost certainly uses a different scheme (a bare `user_id`, `"global"`, `"project--{id}"`, or arbitrary strings) — so migration is fundamentally about **group_id**.

### What you can export
Your Neo4j has `Episodic`/`Entity`/`Community` nodes + `RELATES_TO` edges, each carrying a `group_id`. Raw episode text lives in `EpisodicNode.content` (`nodes.py:318`), readable via `EpisodicNode.get_by_group_ids(driver, [gid], limit, uuid_cursor)` or `retrieve_episodes`. **Critical:** if your Graphiti was built with `store_raw_episode_content=False`, `add_episode` blanked the text (`graphiti.py:665-666`) — then **re-ingest (Strategy B) is impossible**; only remap (A) works.

### Strategy A — remap group_ids in place (fast, lossless, graph-only)
Point NS's `NEO4J_*` at your existing database and rewrite `group_id` to NS's format. NS ships the tool: **`neuralscape-service/scripts/migrate_graph_groups.py`** (dry-run by default; `--apply` to commit; correct labels `Episodic`/`Entity`/`EntityEdge`/`Community`).

```cypher
// remap source group $old → NS private namespace for $owner
MATCH (n) WHERE n.group_id = $old AND any(l IN labels(n) WHERE l IN ['Episodic','Entity','Community'])
  SET n.group_id = 'user--' + $owner;
MATCH ()-[r:RELATES_TO]->() WHERE r.group_id = $old
  SET r.group_id = 'user--' + $owner;
// optional: seed NS props so wiki/back-ref features don't choke (search doesn't need them)
MATCH (n) WHERE n.group_id = 'user--' + $owner AND n.ns_owner IS NULL
  SET n.ns_owner = $owner, n.ns_visibility = 'private';
```

- **Works immediately for graph recall** — NS graph search filters only on `group_id`; missing `memory_id`/`wiki_path`/`ns_*` props don't break recall (they only power the wiki/back-reference features).
- **Lossless** — preserves timestamps, dedup state, and edge-invalidation history.
- **Idempotent** — re-runs only match rows still under `$old`.
- **Graph-only** — there are no Qdrant vectors for this data, so NS hybrid search returns graph triples but no vector hits until you also do Strategy B.
- ⚠️ **Do not** use the older `scripts/migrate-group-ids.cypher` — it matches `EpisodicNode`/`EntityNode` (wrong labels; real ones are `Episodic`/`Entity`), so it's a no-op. Always **back up Neo4j and dry-run first.**

### Strategy B — re-ingest episodes (full NS experience)
Pull each `EpisodicNode.content` and replay it through `POST /v1/memories` (the LLM path that lands in **both** Qdrant and Neo4j with NS scoping + categorization + `attach_memory_id`).

```python
import httpx
from graphiti_core.nodes import EpisodicNode

async def export(driver, src_gid):
    cur, out = None, []
    while (b := await EpisodicNode.get_by_group_ids(driver, [src_gid], limit=100, uuid_cursor=cur)):
        out += b; cur = b[-1].uuid
    return out

def reingest(eps, base, token, user_id, project_id=None):
    with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {token}"}) as c:
        for ep in eps:
            if not ep.content: continue          # store_raw_episode_content was False → skip
            c.post("/v1/memories", json={"messages":[{"role":"user","content":ep.content}],
                                         "user_id":user_id, "project_id":project_id})
```

- **What's lost:** original timestamps (`reference_time=now`), existing dedup/invalidation history (NS re-derives entities/edges from scratch — may differ from the originals), and LLM cost per episode.
- **Not idempotent** — track processed episode UUIDs to avoid duplicate `Episodic` nodes.

### Recommendation matrix

| Situation | Strategy |
|---|---|
| Graph-only, no vector search needed | **A** — fast, lossless, preserves history |
| Want full NS (vector + categories + wiki) | **B** — only B populates Qdrant + applies NS scoping |
| Small graph (≲ few thousand episodes) | **B** — re-extraction cost tolerable |
| Large graph | **A** now, optional **B** on high-value subsets later |
| Built with `store_raw_episode_content=False` | **A only** — raw text was discarded |
| Source uses legacy NS group_ids (`global`, `project--{id}`) | **A** via `migrate_graph_groups.py` as-is |
| Keep history *and* get vectors | **A then B** — remap for continuity, re-ingest key episodes (accept duplicate Episodic nodes) |

---

## After migrating
- Verify with `POST /v1/search` (hybrid vector+graph) and `POST /v1/graph/search` (graph only); spot-check `source_type=imported` rows landed with expected category/visibility.
- Decide visibility deliberately: import private by default, then promote to shared with the team-pool tooling (`bulk_promote_visibility.py`) rather than importing straight to `shared`.
- mem0 imports are vector-first (NS regrows the graph); Graphiti Strategy-A imports are graph-only (no vectors). Run both paths if you're consolidating a mem0 + Graphiti setup into one NS deployment.
