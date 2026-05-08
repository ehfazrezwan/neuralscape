---
title: LLM Extraction & Prompts
date: 2026-05-06
tags: [reference, neuralscape, gemini, prompts, embeddings]
source: handwritten
---

# LLM Extraction & Prompts

## Overview

The extraction layer is what converts free-form conversation messages into structured, categorised memory rows. It is intentionally minimal: a single Gemini call against one master prompt, a JSON-first response parser with a regex fallback, a junk filter that drops raw tool-log noise, and a batched embed-and-upsert into Qdrant. Graph re-ingestion via Graphiti runs once per conversation, not per fact. There is no prompt versioning subdirectory, no function-calling schema, and no JSON-mode flag forced on the model — Gemini is steered purely by prompt engineering. The whole pipeline lives in two files: [`neuralscape-service/prompts.py`](../../neuralscape-service/prompts.py) for the prompt and parsers, and [`neuralscape-service/memory_service.py`](../../neuralscape-service/memory_service.py) for the Gemini client wiring, retry helper, junk filter, and batch storage. Failures are absorbed (an extraction error returns `[]` rather than raising), so the async write pipeline degrades gracefully when the LLM is unavailable.

## The extraction prompt

There is exactly one extraction prompt in the codebase: `CODING_ASSISTANT_EXTRACTION_PROMPT` at `neuralscape-service/prompts.py:25-66`. It is a single triple-quoted string. The first lines tell the model what it is doing and what shape the output should take:

```text
You are a memory extraction engine for an AI coding assistant.

Analyze the conversation below and extract distinct, factual memories
about the user, their preferences, projects, and technical environment.

Each extracted fact MUST be prefixed with a category tag in square brackets.
Use ONLY these categories:
```

After that introduction the prompt embeds all 13 categories inline as a bulleted list — `preference`, `personal_fact`, `technical_skill`, `domain_knowledge`, `tech_stack`, `convention`, `architecture`, `dependency`, `decision`, `interaction`, `workflow`, `procedure`, `task_context`. Each category gets a one-line gloss so the model can disambiguate (for example `[architecture]` is described as "Design decisions, module boundaries, API patterns" while `[convention]` is "Coding conventions, naming, file structure"). The category list is duplicated here from `schemas.MEMORY_CATEGORIES`; the enum is the source of truth for parser validation, but the prompt copy is what the model actually sees. See [03-memory-model](./03-memory-model.md) for how those categories map to scopes.

The prompt then issues eight rules. Rules 1-6 are about quality (factual only, standalone sentences, specificity, dedup, pick the most specific category, mention project name when known). Rules 7-8 (lines 52-53) are the junk-filter rules and they are unusually emphatic:

```text
7. NEVER extract raw tool operations, shell commands run, files
   edited/read/written, git operations, terminal output, or build/test
   execution logs — these are ephemeral actions, not reusable knowledge.
8. NEVER extract information only meaningful in the current session
   context (e.g., "currently running tests", "just fixed a bug in X file").
```

These two rules exist because the upstream conversation feed for an AI coding assistant is dominated by tool-call traces. Without explicit instruction the model happily turns "Ran command: pytest" into a "fact". Rules 7-8 are the first line of defence; the regex junk filter in `memory_service.py` is the second.

The prompt closes by demanding a JSON envelope of the form `{"facts": ["[category] fact", ...]}` and explicitly permits `{"facts": []}` when nothing memorable is present. The conversation messages are appended verbatim after the prompt by `build_extraction_messages()` (`prompts.py:123-144`), which simply concatenates `role: content\n` lines.

## Response parsing

`parse_extraction_response()` (`prompts.py:92-120`) is the only consumer of the LLM's text output. It does three things in order:

1. **Strip markdown code-block wrappers.** Gemini occasionally wraps its JSON in ```json … ``` even when not asked. Lines 104-107 strip a leading ```` ```json ```` (or just ```` ``` ````) and a trailing ```` ``` ```` before the JSON loader sees it.
2. **Try `json.loads()`.** If the response is valid JSON the `facts` key is read directly (line 109-110).
3. **Regex fallback.** If `json.loads()` raises `JSONDecodeError` (or the response was somehow not a string), the parser walks the response line-by-line and keeps every line that starts with a `[category]` bracket (lines 111-118). This is a pragmatic salvage path — Gemini sometimes returns a markdown bullet list or a half-broken JSON object, and the regex fallback recovers most of the content.

Each surviving line is then handed to `parse_category_from_fact()` (`prompts.py:69-89`). That function applies the regex `^\[(\w+)\]\s*(.+)$` and validates the captured category against `MEMORY_CATEGORIES`. Three cases:

- **Valid category** — return `(category_lower, content)`.
- **Bracket present but unknown category** — log a warning at line 86 (`Unknown category '<x>' in fact, defaulting to personal_fact`) and emit `("personal_fact", content)`.
- **No bracket at all** — silently treat the whole string as a `personal_fact`.

The default-to-`personal_fact` policy means the pipeline never throws on a malformed extraction; the worst-case outcome is a fact stored under a generic category rather than dropped.

## Junk filtering

Rules 7-8 in the prompt are reinforced by a deterministic post-filter in `memory_service.py`. The patterns live at `memory_service.py:32-43`:

```python
_JUNK_PATTERNS = [
    r"^Ran command:",
    r"^Edited file[:\s]",
    r"^Wrote file[:\s]",
    r"^Read file[:\s]",
    r"^Created file[:\s]",
    r"^Deleted file[:\s]",
    r"^Launched \w+ task:",
    r"^Tool result:",
    r"^Command output:",
]
_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE | re.MULTILINE)
```

`_is_junk_fact()` (`memory_service.py:49-54`) drops a fact if its stripped length is below 10 characters or if `_JUNK_RE` matches anywhere inside it. The 10-char minimum is a cheap way to catch one-word artefacts ("yes", "ok", file extensions) that slip past both the prompt and the regex.

The filter is applied at `memory_service.py:317-325`, immediately after parsing. Pre-filter and post-filter counts are captured so the log line `"Filtered N junk facts from extraction"` shows up whenever anything is dropped. In practice this removes 5-10% of raw extractions on a typical coding session.

The same `_JUNK_RE` is also reused by `_clean_conversation_for_graph()` (`memory_service.py:57-76`) to strip junk lines from the conversation text before it is handed to Graphiti. Note this filtering happens **before** graph ingestion only — the LLM extraction call sees the unfiltered conversation, on the assumption that the prompt rules and the post-filter together are sufficient.

## Gemini configuration

The Gemini wiring lives in `config.py` and `memory_service.py`. The relevant settings (`config.py:11-13`) are:

- `gemini_llm_model = "gemini-3-flash-preview"` — primary model for extraction
- `gemini_llm_fallback_model = "gemini-2.5-flash"` — used after primary exhausts its retries
- `gemini_embedder_model = "gemini-embedding-001"` — used for both writes and read-time queries

LLM sampling parameters come from mem0's `BaseLlmConfig` (`mem0/configs/llms/base.py:19-22`): temperature 0.1, max_tokens 2000, top_p 0.1, top_k 1. These are aggressively deterministic — the goal is repeatable extraction, not creative paraphrasing.

The Gemini client is built once and cached. `_get_genai_client()` (`memory_service.py:252-261`) lazy-initialises a `genai.Client(api_key=settings.google_api_key)` under the same `_init_lock` that protects the mem0 `Memory` singleton, so concurrent extraction calls share one client.

The actual API call lives at `memory_service.py:298-310`:

```python
response = retry_transient(
    client.models.generate_content,
    model=settings.gemini_llm_model,
    contents=extraction_messages[0]["content"],
    config=GenerateContentConfig(
        http_options=HttpOptions(timeout=60_000),  # ms
    ),
    operation="LLM extraction",
    fallback_model=settings.gemini_llm_fallback_model,
)
```

Three things to note. First, the call is wrapped by `retry_transient` (see below). Second, the HTTP timeout is 60 seconds — long enough for a large conversation but short enough that a stuck request fails the ARQ task instead of hanging the worker. Third, there is no `response_mime_type="application/json"` and no function-calling schema; the JSON envelope is enforced entirely by the prompt and salvaged by the regex fallback when the model misbehaves.

## Embeddings

Embeddings use Gemini's `gemini-embedding-001` at 768 dimensions. The dimension is hard-coded in two places in the mem0 config (`config.py:84` for Qdrant, `config.py:105` for the embedder), and `mem0/embeddings/gemini.py:15-16` negotiates it as `embedding_dims or output_dimensionality or 768`.

There are three embedding code paths:

- **Write path (batch).** After extraction and junk filtering, `_batch_store_facts` issues a single `m.embedding_model.embed_batch(texts, memory_action="add")` call (`memory_service.py:545-546`) for every fact in the conversation, followed by one Qdrant upsert at line 549. This is the cost-critical path: N facts cost one HTTP round trip, not N.
- **Read path (per query).** `MemoryService.search()` delegates to `m.search()`, which embeds the query string itself (`memory_service.py:625`). One query, one embedding call.
- **Dedup path (per memory).** The dedup cron embeds each candidate memory individually via `m.embedding_model.embed(text)` (`memory_service.py:1568`). This is acceptable because dedup runs at most every 6 hours and operates on a bounded batch.

See [06-storage-backends](./06-storage-backends.md) for how those vectors are persisted into Qdrant.

## mem0 factory wiring

The `Memory.from_config()` constructor consumes the dict built by `Settings.get_mem0_config()` at `config.py:79-132`. Two relevant blocks:

```python
"llm": {
    "provider": "gemini",
    "config": {
        "model": settings.gemini_llm_model,
        "api_key": settings.google_api_key,
    },
},
"embedder": {
    "provider": "gemini",
    "config": {
        "model": settings.gemini_embedder_model,
        "api_key": settings.google_api_key,
        "embedding_dims": 768,
    },
},
```

`provider: "gemini"` resolves through mem0's factory (`mem0/utils/factory.py:48` for LLMs, `:145` for embedders) to `mem0.llms.gemini.GeminiLLM` and `mem0.embeddings.gemini.GoogleGenAIEmbedding`. The graph store block also receives Gemini wiring — `graphiti_llm_provider`, `graphiti_llm_model`, `graphiti_llm_fallback_model`, `graphiti_embedder_model`, and `graphiti_reranker_provider` — so Graphiti uses the same model family for entity extraction inside the knowledge graph. The CLAUDE.md note at the project root warns that after upstream syncs `mem0/configs.py` and `mem0/utils/factory.py` should be checked for provider-registration conflicts.

## Retry strategy

`retry_transient()` (`memory_service.py:97-157`) is the unified retry helper for every Gemini call in the service — extraction, embedding, and graph add all use it. Its logic:

1. **Transient detection by message substring.** `_is_transient()` (`memory_service.py:25-28`) scans the exception's stringified message for any of `503`, `429`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`, `rate limit`, `overloaded`, `capacity`, `timed out`, `timeout`. Anything else is re-raised immediately so genuine bugs (auth failures, schema errors) propagate.
2. **Exponential backoff with jitter.** On a transient error, sleep for `min(base * 2 ** attempt + uniform(0, 1), max_delay)` (line 133). Defaults from `config.py:38-40` are 3 retries, 1.0 s base, 30 s cap. Each retry logs a warning with the attempt counter and the upcoming delay.
3. **Fallback model.** If the primary model exhausts its retry budget on transient errors and a `fallback_model` was passed, the helper swaps `kwargs["model"]` to the fallback and tries one final time (lines 141-155). Extraction passes `fallback_model=settings.gemini_llm_fallback_model` so a degraded `gemini-3-flash-preview` falls through to the stable `gemini-2.5-flash`.

## Failure modes

The extraction pipeline has three distinct failure paths and each one is contained:

- **LLM error (any cause).** The `try/except` at `memory_service.py:298-314` catches every exception from `retry_transient` and from `parse_extraction_response`. It logs an `error` and returns `[]`. The caller sees zero extracted facts but the request does not fail.
- **Malformed JSON.** Handled inside `parse_extraction_response` via the regex line-by-line fallback. If the regex also recovers nothing, the function returns `[]` without raising.
- **Non-transient API error.** `retry_transient` re-raises immediately on the first attempt, which is then caught by the outer try/except and converted to `[]`. There is no second-chance prompt and no alternative extraction strategy.

Graph ingestion failures are similarly non-fatal — `memory_service.py:363-364` logs a warning and continues, so a Neo4j outage cannot block a vector write.

## Cost optimisations

Several deliberate choices keep token and call costs predictable:

- **Single extraction call per conversation.** One Gemini request emits all facts for a whole turn, regardless of how many facts come back.
- **Single batch embed.** `embed_batch` is one HTTP round trip for N facts, not N round trips.
- **Single Qdrant upsert.** All embeddings land in one upsert so vector-store contention is minimal.
- **Junk filter before storage.** Roughly 5-10% of extracted "facts" are dropped before they ever consume embed quota or vector-store space.
- **Single graph re-ingestion.** Graphiti gets the cleaned conversation once per turn (`memory_service.py:356-362`), not once per fact.
- **Dedup cron.** A 6-hour cron with a 0.95 cosine threshold (`config.py:43, 45`) keeps long-term storage from drifting upward.
- **60 s LLM timeout.** Bounds worst-case worker latency.

There is no prompt versioning, no `prompts/v1` subdirectory, and no AB harness — the prompt and junk patterns are edited inline. The presence of both rule 7 in the prompt and `_JUNK_RE` in code suggests the system has been hardened against prior overfitting to tool logs; the regex fallback in `parse_extraction_response` similarly looks like a response to observed Gemini output variability. See [07-async-pipeline](./07-async-pipeline.md) for how this extraction call is invoked from the ARQ worker.

## Related

- [04-memory-service-core](./04-memory-service-core.md)
- [03-memory-model](./03-memory-model.md)
- [06-storage-backends](./06-storage-backends.md)
- [07-async-pipeline](./07-async-pipeline.md)
- [02-service-architecture](./02-service-architecture.md)
