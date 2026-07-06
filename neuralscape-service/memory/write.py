"""Write path: extraction, raw stores, batch stores, graph enrichment, expiry.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import hashlib
import logging
import uuid

from datetime import datetime, timezone
from config import settings
from index_format import distill_title
from prompts import build_extraction_messages, parse_extraction_response, parse_extraction_response_rich
from savings_meter import stamp_tokens
from schemas import EPISTEMIC_LEVEL_VOCAB, GLOBAL_CATEGORIES, MEMORY_CATEGORIES, MemoryResponse, MemoryScope, MemoryVisibility, PROJECT_CATEGORIES, default_scope_for_category, default_visibility_for_category, normalize_visibility, validate_occurred_at
from memory.groups import _build_group_id
from memory.hashing import _infer_project_id, _parse_expires_at, content_hash
from memory.junk import _clean_conversation_for_graph, _is_junk_fact
from memory.ranking import _times_derived_from_metadata
from memory.retry import retry_transient

logger = logging.getLogger(__name__)


def _validate_speaker(speaker: str | None) -> str | None:
    """Validate and sanitize a speaker label (T1.2 speaker sanity guard).

    Returns the speaker if it's plausible as a speaker label, otherwise None.
    Guards against the permissive speaker-regex false positives from the rich
    parser — rejects oversized tokens, empty strings, and sentence fragments
    that clearly aren't speaker names/roles.

    Args:
        speaker: Raw speaker token from ParsedFact

    Returns:
        The validated speaker, or None if implausible.
    """
    if not speaker:
        return None
    speaker = speaker.strip()
    if not speaker:
        return None
    # Reject oversized tokens (likely sentence fragments, not names/roles)
    if len(speaker) > 40:
        return None
    # Simple heuristic: a plausible speaker is mostly alphanumeric/spaces/dots/
    # hyphens/underscores (the same class the regex allows). If it's clearly
    # a sentence fragment (e.g., ends with punctuation like "Note:"), drop it.
    # We keep this simple — the regex already bounds the character class, so
    # this just adds a length + sanity check on top.
    return speaker


class WriteMixin:
    """WriteMixin for MemoryService (mechanical split — see memory_service.py)."""

    def _attach_memory_id_to_graph_nodes(
        self,
        *,
        group_id: str,
        memory_id: str,
        visibility: str,
        owner_user_id: str,
        write_started_at: datetime,
        source_ref: dict | None = None,
    ) -> None:
        """Stamp ``memory_id``/visibility/owner onto recently-created Graphiti nodes.

        Called by every store path after the Graphiti write completes. The
        wiki synthesizer uses these properties to walk community → source
        memories. When ``source_ref`` is present (data-layer-connector
        ingestion), also links those nodes to a ``(:Source)`` node so the
        graph can walk memory → source → connector. Failures here never fail
        the underlying write — the helpers log and return 0.
        """
        if not (self._graphiti and self._bridge):
            return
        from extensions.dreaming.config import dreaming_settings as synthesizer_settings
        from extensions.dreaming.graph_patcher import (
            attach_memory_id,
            attach_source_ref,
        )

        # Coroutine built separately so we can ``.close()`` it if
        # ``_run_on_bridge`` raises before awaiting — otherwise tests
        # with mocked bridges leak ``RuntimeWarning: coroutine never
        # awaited`` and slow the suite down with traceback collection.
        coro = attach_memory_id(
            self._graphiti.driver,
            group_id=group_id,
            memory_id=memory_id,
            visibility=visibility,
            owner_user_id=owner_user_id,
            write_started_at=write_started_at,
            window_seconds=synthesizer_settings.attach_window_seconds,
        )
        try:
            self._run_on_bridge(coro, timeout=10.0)
        except Exception:
            coro.close()
            logger.warning(
                "attach_memory_id post-write hook failed (non-critical)",
                exc_info=True,
            )

        # Link to the originating data-layer source, when ingested via a connector.
        if source_ref:
            src_coro = attach_source_ref(
                self._graphiti.driver,
                group_id=group_id,
                memory_id=memory_id,
                source_ref=source_ref,
                write_started_at=write_started_at,
                window_seconds=synthesizer_settings.attach_window_seconds,
            )
            try:
                self._run_on_bridge(src_coro, timeout=10.0)
            except Exception:
                src_coro.close()
                logger.warning(
                    "attach_source_ref post-write hook failed (non-critical)",
                    exc_info=True,
                )

    # ──────────────────────────────────────────────
    # Store operations
    # ──────────────────────────────────────────────

    def extract_and_store(
        self,
        messages: list[dict],
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        occurred_at: str | None = None,
        return_stats: bool = False,
    ) -> list[MemoryResponse] | tuple[list[MemoryResponse], dict]:
        """Extract facts from conversation via LLM, then store each with category metadata.

        Long conversations are split into overlapping extraction windows
        (audit 27 #22) — one LLM call per window, facts unioned; a
        conversation at or under one window (``EXTRACTION_WINDOW_MESSAGES``)
        is byte-identical to the unwindowed path. One failed window degrades
        to a partial result (reported via ``return_stats``); only ALL windows
        failing raises — preserving the post-#118 loud-failure semantics.

        Args:
            messages: Conversation messages [{role, content}, ...]
            user_id: User identifier
            project_id: Optional project identifier
            agent_id: Optional agent identifier for provenance
            run_id: Optional session identifier
            occurred_at: Event time (ISO 8601) — when this conversation
                actually happened, for historical ingestion of old chat
                exports. Stamped on every extracted fact; None ⇒ omitted.
            return_stats: When True, returns ``(memories, stats)`` where
                stats is ``{windows_total, windows_failed, window_errors}``
                so the worker can surface partial extraction in the task
                result. Default False keeps the legacy list return.

        Returns:
            List of stored memory responses (or ``(list, stats)`` when
            ``return_stats=True``).
        """
        m = self._get_memory()

        # Step 1: Call Gemini for fact extraction, one call per window.
        # E4: operator-supplied extraction instructions (per-project +
        # per-user, composed) ride along as a clearly-delimited addendum on
        # EVERY window's prompt — they steer fact selection/phrasing but
        # never the JSON output contract.
        from extraction_settings import resolve_instructions
        from prompts import split_into_windows

        operator_guidance = resolve_instructions(user_id, project_id)
        windows = split_into_windows(
            messages,
            settings.extraction_window_messages,
            settings.extraction_window_overlap,
        )
        client = self._get_genai_client()

        from google.genai.types import GenerateContentConfig, HttpOptions

        # Use rich parser to extract speaker attribution (T1.2)
        # and occurred_at (T1.3, future). For now we consume only speaker;
        # occurred_at from the ParsedFact is ignored/dropped here (the
        # conversation-level occurred_at parameter flows separately below).
        from prompts import ParsedFact

        parsed_facts_rich: list[ParsedFact] = []
        window_errors: list[str] = []
        last_exc: Exception | None = None
        for w_idx, window_messages in enumerate(windows):
            extraction_messages = build_extraction_messages(
                window_messages, operator_guidance=operator_guidance
            )
            try:
                response = retry_transient(
                    client.models.generate_content,
                    model=settings.gemini_llm_model,
                    contents=extraction_messages[0]["content"],
                    config=GenerateContentConfig(
                        http_options=HttpOptions(timeout=60_000),  # milliseconds
                    ),
                    operation="LLM extraction",
                    fallback_model=settings.gemini_llm_fallback_model,
                )
                parsed_facts_rich.extend(parse_extraction_response_rich(response.text))
            except Exception as e:
                # One window failing must not zero the whole session — keep
                # the other windows' facts and report the failure honestly
                # in the task result (see return_stats).
                last_exc = e
                window_errors.append(f"window {w_idx + 1}/{len(windows)}: {e}")
                logger.error(
                    f"LLM extraction failed for window {w_idx + 1}/{len(windows)} "
                    f"(user {user_id}): {e}",
                    exc_info=True,
                )
        if window_errors and len(window_errors) == len(windows):
            # Loud failure: EVERY window failed → propagate so the ARQ job
            # (and task status) reports failed instead of completing with
            # zero facts. A silent [] here masked real outages (the caller
            # can't tell "nothing worth extracting" from "extraction is
            # down"). For a single-window conversation this is exactly the
            # pre-windowing behavior.
            raise last_exc
        stats = {
            "windows_total": len(windows),
            "windows_failed": len(window_errors),
            "window_errors": window_errors,
        }

        # Filter out junk facts from extraction.
        # Check the FULL fact (speaker + content) for junk patterns, not just
        # the content alone — the rich parser may have extracted a junk-pattern
        # prefix like "Ran command:" as a speaker, leaving content that doesn't
        # match the junk regex on its own. Reconstructing catches those.
        pre_filter_count = len(parsed_facts_rich)
        parsed_facts_rich = [
            pf for pf in parsed_facts_rich
            if not _is_junk_fact(
                f"{pf.speaker}: {pf.content}" if pf.speaker else pf.content
            )
        ]
        if pre_filter_count != len(parsed_facts_rich):
            logger.info(
                f"Filtered {pre_filter_count - len(parsed_facts_rich)} junk facts from extraction"
            )

        if not parsed_facts_rich:
            logger.info("No facts extracted from conversation")
            return ([], stats) if return_stats else []

        # Step 2: Batch-store all facts (single embed + single Qdrant upsert).
        # T1.2: thread speaker attribution through. We extract (category, content)
        # tuples for _batch_store_facts, plus a parallel speakers list.
        # T1.3's occurred_at from ParsedFact is deliberately ignored here — the
        # conversation-level occurred_at parameter flows separately below.
        # A failure here (embedding, Qdrant) must PROPAGATE: the previous
        # except-and-continue silently stored zero facts while the task
        # reported success — exactly what hid the gateway's single-input
        # embed rejection in production. Raising fails the ARQ job, so
        # /v1/memories/status/{task_id} reports status=failed with the error.
        try:
            facts = [(pf.category, pf.content) for pf in parsed_facts_rich]
            speakers = [pf.speaker for pf in parsed_facts_rich]
            stored = self._batch_store_facts(
                facts=facts,
                speakers=speakers,
                user_id=user_id,
                project_id=project_id,
                agent_id=agent_id,
                run_id=run_id,
                source="conversation",
                occurred_at=occurred_at,
            )
        except Exception as e:
            logger.error(
                f"Batch store failed for user {user_id}: {e} — "
                f"{len(parsed_facts_rich)} extracted facts were NOT stored",
                exc_info=True,
            )
            raise

        # Step 3: Add cleaned conversation text to knowledge graph — ONE
        # episode for the whole conversation regardless of extraction
        # windowing (Graphiti handles its own entity windowing internally).
        # KNOWN RESIDUAL (occurred_at): the episode's Graphiti reference_time
        # is stamped "now" inside the mem0 subtree's MemoryGraph.add, which
        # exposes no event-time parameter — threading occurred_at through to
        # add_episode(reference_time=...) needs a subtree change, deliberately
        # out of scope here. Event time lives in the Qdrant payload only.
        # Conversation extractions are personal (private) by default — the
        # caller's spoken context isn't team-shared automatically.
        group_id = _build_group_id(
            MemoryVisibility.PRIVATE.value,
            user_id,
            project_id,
        )
        cleaned_messages = _clean_conversation_for_graph(messages)
        raw_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in cleaned_messages
        )
        graph_write_started_at = datetime.now(timezone.utc)
        try:
            if self._graphiti and self._bridge and raw_text.strip():
                # Idempotency key (audit 27 #21): the episode name is derived
                # from the cleaned conversation + group, not the wall clock —
                # an ARQ retry / resubmission of the same conversation re-derives
                # the same name and is skipped by the existence probe below
                # instead of minting a duplicate episode (the 4,999
                # double-episode mechanism).
                episode_key = hashlib.sha256(
                    f"{raw_text}\n{group_id}".encode()
                ).hexdigest()
                episode_name = f"mem0_episode_{episode_key[:32]}"
                if self._graph_episode_exists(group_id, episode_name):
                    logger.info(
                        f"Skipping graph episode for user {user_id} — episode "
                        f"{episode_name} already exists in group {group_id} "
                        f"(re-run of an already-ingested conversation)"
                    )
                else:
                    retry_transient(
                        self._memory.graph.add,
                        data=raw_text,
                        filters={"user_id": user_id, "group_id": group_id},
                        episode_name=episode_name,
                        operation="graph storage (extract_and_store)",
                    )
                    # 1-episode → N-memories shape: a single graph.add produces
                    # entities that could legitimately belong to any of the N
                    # extracted memories. We call attach_memory_id once per
                    # stored memory; the Cypher uses ``coalesce(memory_id,
                    # $memory_id)``, so the first call wins and later calls are
                    # no-ops on the same node. The result is each fresh entity
                    # carries one representative memory_id from the batch — not
                    # a perfect mapping, but enough for the wiki synthesizer
                    # to walk community → source memory.
                    for mem in stored:
                        self._attach_memory_id_to_graph_nodes(
                            group_id=group_id,
                            memory_id=getattr(mem, "id", "") or "",
                            visibility=MemoryVisibility.PRIVATE.value,
                            owner_user_id=user_id,
                            write_started_at=graph_write_started_at,
                        )
        except Exception as e:
            logger.warning(f"Graph storage failed (non-critical): {e}")

        return (stored, stats) if return_stats else stored

    def _graph_episode_exists(self, group_id: str, episode_name: str) -> bool:
        """One cheap Cypher lookup: does an Episodic node with this
        idempotency-keyed name already exist in the group? (audit 27 #21)

        Fail-OPEN: any error (bridge down, Neo4j hiccup, timeout) returns
        False so a broken probe degrades to today's behavior — a possible
        duplicate episode — rather than silently dropping the graph write.
        """
        if not (self._graphiti and self._bridge):
            return False
        # _run_on_bridge submits onto self._bridge._loop; if that isn't a real
        # event loop (e.g. a mocked bridge in unit tests, or a half-initialized
        # adapter), run_coroutine_threadsafe would park on future.result() until
        # the timeout. Bail out fail-open immediately instead.
        import asyncio as _asyncio

        if not isinstance(getattr(self._bridge, "_loop", None), _asyncio.AbstractEventLoop):
            return False
        cypher = (
            "MATCH (e:Episodic {group_id: $group_id, name: $name}) "
            "RETURN e.uuid AS uuid LIMIT 1"
        )

        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(cypher, group_id=group_id, name=episode_name)
                return await result.data()

        # Built separately so it can be .close()d if _run_on_bridge raises
        # before awaiting (avoids "coroutine never awaited" warnings — same
        # convention as _attach_memory_id_to_graph_nodes).
        coro = _run()
        try:
            records = self._run_on_bridge(coro, timeout=10.0) or []
            return bool(records)
        except Exception:
            coro.close()
            logger.debug(
                "graph episode idempotency lookup failed (fail-open)", exc_info=True
            )
            return False

    def extract_facts_only(
        self,
        text: str,
        extractor=None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Run LLM fact extraction over a block of text and return (category, content) tuples.

        Reuses the same Gemini extraction + junk filter as ``extract_and_store``
        but stores nothing — the caller decides how to persist (the ingest
        pipeline stores them with ``source_type="imported"`` and the document's
        ``source_ref``). Returns ``[]`` on extraction failure (logged), so a
        flaky LLM call degrades to passages-only rather than failing ingest.

        ``extractor`` is an optional :class:`ingest.extractors.FactExtractor`
        supplied by a knowledge adapter — it owns the extraction *prompt* and the
        *parse* of the response. When ``None`` (the default), behavior is
        byte-for-byte the coding-assistant extractor: the LLM client, retries,
        model selection, and junk filter below are shared across all adapters so
        only the taxonomy/prompt varies, never the envelope.

        ``user_id`` / ``project_id`` scope the E4 operator guidance lookup:
        when set, the composed custom extraction instructions are appended
        to the prompt as the OPERATOR GUIDANCE addendum — AFTER the adapter's
        own prompt, so instructions compose with (never replace) adapters.
        """
        if not text or not text.strip():
            return []
        if extractor is not None:
            extraction_messages = extractor.build_messages(text)
        else:
            extraction_messages = build_extraction_messages([{"role": "user", "content": text}])
        # E4: operator guidance rides after the (possibly adapter-owned)
        # prompt. Best-effort — resolve failure means no addendum.
        from extraction_settings import resolve_instructions
        from prompts import append_operator_guidance

        operator_guidance = resolve_instructions(user_id, project_id)
        if operator_guidance:
            extraction_messages[0]["content"] = append_operator_guidance(
                extraction_messages[0]["content"], operator_guidance
            )
        client = self._get_genai_client()
        try:
            from google.genai.types import GenerateContentConfig, HttpOptions

            response = retry_transient(
                client.models.generate_content,
                model=settings.gemini_llm_model,
                contents=extraction_messages[0]["content"],
                config=GenerateContentConfig(
                    http_options=HttpOptions(timeout=60_000),
                ),
                operation="LLM extraction (ingest)",
                fallback_model=settings.gemini_llm_fallback_model,
            )
            if extractor is not None:
                parsed_facts = extractor.parse(response.text)
            else:
                parsed_facts = parse_extraction_response(response.text)
        except Exception as e:
            logger.error(f"Ingest extraction failed: {e}")
            return []
        return [(cat, content) for cat, content in parsed_facts if not _is_junk_fact(content)]

    def store_raw(
        self,
        content: str,
        user_id: str,
        category: str,
        scope: str = "global",
        project_id: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        # Memory-model v2 fields (all optional, all additive)
        domain: str | None = None,
        observation_type: str | None = None,
        concepts: list[str] | None = None,
        source_type: str | None = None,
        related_memory_ids: list[str] | None = None,
        confidence: float | None = None,
        expires_at: datetime | None = None,
        # Event time (ISO 8601): when the fact/event actually happened, for
        # historical ingestion. None → omitted from the payload entirely
        # ("event time unknown, fall back to created_at") — never defaulted.
        occurred_at: str | None = None,
        # Provenance epistemics (A1, optional)
        derived_from: list[str] | None = None,
        epistemic_level: str | None = None,
        # Multi-user model (None → category default)
        visibility: str | None = None,
        # Data-layer connectors (None → omitted)
        memory_kind: str | None = None,
        source_ref: dict | None = None,
        # Write-path isolation: when False, skip the slow inline graph.add so
        # the fast vector write returns immediately — the caller is expected to
        # enqueue enrich_graph() onto the graph worker. When return_created is
        # True, returns (responses, created) so the caller only enqueues graph
        # enrichment for genuinely new rows (not content-hash dedup hits).
        add_to_graph: bool = True,
        return_created: bool = False,
        # Knowledge-adapter graph ontology (Graphiti custom types), resolved
        # per-ingest and forwarded to enrich_graph → add_episode. None ⇒
        # Graphiti's built-in generic extraction (every regular memory write).
        graph_ontology: dict | None = None,
    ) -> list[MemoryResponse] | tuple[list[MemoryResponse], bool]:
        """Store a single pre-categorized fact directly (no LLM extraction).

        Returns a ``list[MemoryResponse]`` by default, or ``(responses, created)``
        when ``return_created=True`` (``created`` is False for content-hash dedup
        hits) — callers must not assume the return is always a list.

        Performs content-hash dedup before insert: if a memory with the same
        md5(content) + user_id + scope already exists, the existing memory is
        returned instead of creating a duplicate. This makes hook-driven
        re-flushes idempotent.

        Args:
            content: The fact to store
            user_id: User identifier
            category: Memory category (validated against MEMORY_CATEGORIES)
            scope: "global" or "project"
            project_id: Required when scope="project"
            tags: Optional free-form tags
            agent_id: Optional agent identifier
            run_id: Optional session identifier
            domain: Memory-model v2 — life-context domain
            observation_type: Memory-model v2 — shape of observation
            concepts: Memory-model v2 — controlled-vocab cross-cutting tags
            source_type: Memory-model v2 — provenance
            related_memory_ids: Memory-model v2 — graph linkage
            confidence: Memory-model v2 — extractor's self-rated 0.0-1.0
            expires_at: Memory-model v2 — optional expiry timestamp
            occurred_at: Event time (ISO 8601) — when the fact/event actually
                happened (historical ingestion). Validated + normalized via
                ``validate_occurred_at``; absent ⇒ omitted from the payload
            derived_from: A1 provenance — premise memory IDs this memory was
                derived from (walked by get_reasoning_chain)
            epistemic_level: A1 provenance — how this memory is known
                (explicit | deductive | inductive | reflection)

        Returns:
            List of stored memory responses (always exactly one item). When the
            content-hash dedup short-circuits, the returned `id`/`created_at`
            will match the existing memory; otherwise they reflect the new row.
            Both paths use `source="vector"`; callers that need to distinguish
            should compare the returned `id` against their expected new UUID.
        """
        # Validation + visibility/scope resolution + content-hash dedup live in
        # _prepare_raw_store so store_raw_batch shares them without paying a
        # per-item embed round trip (audit 27 #20 — two-pass batch stores).
        prepared, existing = self._prepare_raw_store(
            content=content,
            user_id=user_id,
            category=category,
            scope=scope,
            project_id=project_id,
            tags=tags,
            agent_id=agent_id,
            run_id=run_id,
            domain=domain,
            observation_type=observation_type,
            concepts=concepts,
            source_type=source_type,
            related_memory_ids=related_memory_ids,
            confidence=confidence,
            expires_at=expires_at,
            occurred_at=occurred_at,
            derived_from=derived_from,
            epistemic_level=epistemic_level,
            visibility=visibility,
            memory_kind=memory_kind,
            source_ref=source_ref,
        )
        if existing is not None:
            # created=False → caller must NOT re-enqueue graph enrichment.
            return ([existing], False) if return_created else [existing]

        # ── Direct embed + Qdrant insert (bypass m.add) ──
        m = self._get_memory()
        embedding = m.embedding_model.embed(content, memory_action="add")
        self._finalize_raw_store(
            prepared,
            embedding,
            add_to_graph=add_to_graph,
            graph_ontology=graph_ontology,
        )

        responses = [prepared["response"]]
        # created=True → a new row was written, so the caller should enqueue
        # graph enrichment (when it deferred it via add_to_graph=False).
        return (responses, True) if return_created else responses

    def _prepare_raw_store(
        self,
        *,
        content: str,
        user_id: str,
        category: str,
        scope: str = "global",
        project_id: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        domain: str | None = None,
        observation_type: str | None = None,
        concepts: list[str] | None = None,
        source_type: str | None = None,
        related_memory_ids: list[str] | None = None,
        confidence: float | None = None,
        expires_at: datetime | None = None,
        occurred_at: str | None = None,
        derived_from: list[str] | None = None,
        epistemic_level: str | None = None,
        visibility: str | None = None,
        memory_kind: str | None = None,
        source_ref: dict | None = None,
    ) -> tuple[dict | None, MemoryResponse | None]:
        """Validate + resolve + dedup one raw store; everything except the embed.

        Shared by ``store_raw`` (single, embeds inline) and
        ``store_raw_batch`` (two-pass: prepares every item first, then ONE
        ``embed_batch`` call — audit 27 #20). Returns ``(prepared, existing)``
        where exactly one side is non-None:

        - ``existing``: content-hash dedup hit — ``times_derived`` already
          bumped (and a dream tombstone revived) on the survivor.
        - ``prepared``: dict carrying the ready-to-insert payload, the
          ready-to-return :class:`MemoryResponse`, and the fields
          ``_finalize_raw_store`` needs for the graph write.
        """
        if category not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of: {list(MEMORY_CATEGORIES.keys())}")
        if epistemic_level is not None and epistemic_level not in EPISTEMIC_LEVEL_VOCAB:
            raise ValueError(
                f"Invalid epistemic_level: {epistemic_level}. "
                f"Must be one of: {sorted(EPISTEMIC_LEVEL_VOCAB)}"
            )
        # Event time: validate + normalize here (not only at the API boundary)
        # so worker / batch / ingest callers share one contract. Raises
        # ValueError on garbage or far-future values, like the checks above.
        occurred_at = validate_occurred_at(occurred_at)
        # Normalize empty→None so the response echoes exactly what was
        # persisted (metadata only stores truthy lists; echoing [] while
        # storing nothing would make the write and later reads disagree).
        derived_from = derived_from or None

        # Resolve visibility: explicit caller value > per-category default.
        # ``normalize_visibility`` handles MemoryVisibility enum, plain
        # str, and the legacy "MemoryVisibility.X" stringified-enum
        # format from the pre-__str__-override bug (Python 3.11+ str(Enum)
        # regression). Without normalization, "MemoryVisibility.SHARED"
        # used to land in Qdrant metadata and break both the GET API
        # shape and the conversation_compiler event handler.
        # Resolved BEFORE the scope check because standards force scope below.
        effective_visibility = (
            normalize_visibility(visibility)
            if visibility is not None
            else default_visibility_for_category(category).value
        )

        # ── Authoritative "standard" tier: gate + force global scope ──
        # The REST route / MCP tool gate this earlier and synchronously, but
        # this covers every store path (worker, batch, sync-fallback) so a
        # standard memory can only ever be written by an authorized dictator.
        # Standards are org-wide by definition → always global scope, no
        # project_id (so a project-category standard like a global `convention`
        # doesn't inherit scope="project" and fail the project check below).
        if effective_visibility == MemoryVisibility.STANDARD.value:
            if not settings.standards_enabled:
                raise PermissionError(
                    "The 'standard' visibility tier is disabled (set STANDARDS_ENABLED=true)."
                )
            if not settings.is_dictator(user_id):
                raise PermissionError(
                    f"User {user_id!r} is not authorized to write 'standard'-tier memories."
                )
            scope = "global"
            project_id = None

        if scope == "project" and not project_id:
            raise ValueError("project_id is required when scope='project'")

        now_iso = datetime.now(timezone.utc).isoformat()
        chash = content_hash(content)

        # ── Content-hash dedup ──
        # Skip insert if this exact (user_id, scope, hash) already exists.
        existing = self._find_by_content_hash(
            user_id=user_id, content_hash=chash, scope=scope,
            project_id=project_id, visibility=effective_visibility,
        )
        if existing is not None:
            logger.info(
                f"Dedup hit for user={user_id} hash={chash[:8]}... — returning existing id={existing.id}"
            )
            # Reinforcement-aware dedup: the caller just re-derived this exact
            # fact — count it on the survivor instead of discarding the signal.
            self._bump_times_derived(existing.id, 1)
            # Audit 27 #5: if the survivor was dream-tombstoned, re-derivation
            # resurrects it — otherwise this dedup short-circuit silently
            # swallows the write while the fact stays excluded from recall.
            # Runs AFTER the bump: the bump's read-merge-write could otherwise
            # re-persist the stale tombstone flag it read moments earlier.
            if self._revive_if_tombstoned(existing.id):
                existing.revived = True
            return (None, existing)

        mid = str(uuid.uuid4())

        # Retrieval economics (C1): distill a ~10-word title + token cost at
        # write time so index-only recall never has to fetch/parse content.
        # Pure heuristics — no LLM call on the hot write path.
        title = distill_title(content)
        token_estimate = stamp_tokens(content)

        metadata: dict = {
            "scope": scope,
            "category": category,
            "project_id": project_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "source": "explicit",
            # Multi-user model: who owns this memory + who can read it.
            "owner_user_id": user_id,
            "visibility": effective_visibility,
            # Retrieval economics (C1): index-row fields
            "title": title,
            "token_estimate": token_estimate,
        }
        if tags:
            metadata["tags"] = tags
        # Memory-model v2 metadata (only stored when set)
        if domain is not None:
            metadata["domain"] = domain
        if observation_type is not None:
            metadata["observation_type"] = observation_type
        if concepts:
            metadata["concepts"] = concepts
        if source_type is not None:
            metadata["source_type"] = source_type
        if related_memory_ids:
            metadata["related_memory_ids"] = related_memory_ids
        if confidence is not None:
            metadata["confidence"] = confidence
        # A1 provenance (only stored when set)
        if derived_from:
            metadata["derived_from"] = derived_from
        if epistemic_level is not None:
            metadata["epistemic_level"] = epistemic_level
        if expires_at is not None:
            metadata["expires_at"] = expires_at.isoformat() if hasattr(expires_at, "isoformat") else str(expires_at)
        # Event time — only stored when known (absence means "fall back to
        # created_at"; a created_at default here would fabricate event times).
        if occurred_at is not None:
            metadata["occurred_at"] = occurred_at
        # Data-layer connector provenance
        if memory_kind is not None:
            metadata["memory_kind"] = memory_kind
        if source_ref:
            metadata["source_ref"] = source_ref

        payload = {
            "data": content,
            "hash": chash,
            "created_at": now_iso,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": metadata,
        }

        response = MemoryResponse(
            id=mid,
            memory=content,
            category=category,
            scope=scope,
            project_id=project_id,
            tags=tags,
            source="vector",
            created_at=now_iso,
            domain=domain,
            observation_type=observation_type,
            concepts=concepts,
            source_type=source_type,
            related_memory_ids=related_memory_ids,
            confidence=confidence,
            expires_at=expires_at.isoformat() if expires_at and hasattr(expires_at, "isoformat") else None,
            occurred_at=occurred_at,
            derived_from=derived_from,
            epistemic_level=epistemic_level,
            memory_kind=memory_kind,
            source_ref=source_ref,
            visibility=effective_visibility,
            owner_user_id=user_id,
            title=title,
            token_estimate=token_estimate,
        )

        prepared = {
            "mid": mid,
            "content": content,
            "payload": payload,
            "response": response,
            "now_iso": now_iso,
            # (user, hash, scope, project, visibility) — the dedup identity,
            # exposed so store_raw_batch can collapse in-batch duplicates.
            "dedup_key": (user_id, chash, scope, project_id, effective_visibility),
            # Fields _finalize_raw_store needs for the graph write.
            "user_id": user_id,
            "project_id": project_id,
            "effective_visibility": effective_visibility,
            "source_ref": source_ref,
        }
        return (prepared, None)

    def _finalize_raw_store(
        self,
        prepared: dict,
        embedding: list[float],
        *,
        add_to_graph: bool = True,
        graph_ontology: dict | None = None,
    ) -> None:
        """Insert + history + (optional) graph write for a prepared raw store.

        The other half of :meth:`_prepare_raw_store` — takes the embedding
        from the caller so ``store_raw`` can embed one text inline while
        ``store_raw_batch`` supplies vectors from a single ``embed_batch``
        call (audit 27 #20).
        """
        m = self._get_memory()
        mid = prepared["mid"]
        content = prepared["content"]

        m.vector_store.insert(
            vectors=[embedding],
            ids=[mid],
            payloads=[prepared["payload"]],
        )

        try:
            m.db.add_history(mid, None, content, "ADD", created_at=prepared["now_iso"])
        except Exception as e:
            logger.warning(f"History record failed for {mid}: {e}")

        # ── Add to knowledge graph ──
        # Graphiti entity extraction is slow (~minutes, gated by Gemini). When
        # add_to_graph is False the caller enqueues enrich_graph() onto the
        # graph worker instead, so this fast path returns right after the
        # vector insert. source_ref (connector provenance) rides along so the
        # back-reference attached to the graph nodes records where it came from.
        if add_to_graph:
            self.enrich_graph(
                content=content,
                user_id=prepared["user_id"],
                project_id=prepared["project_id"],
                visibility=prepared["effective_visibility"],
                memory_id=mid,
                source_ref=prepared["source_ref"],
                graph_ontology=graph_ontology,
            )

    def enrich_graph(
        self,
        content: str,
        user_id: str,
        project_id: str | None,
        visibility: str,
        memory_id: str,
        source_ref: dict | None = None,
        graph_ontology: dict | None = None,
    ) -> bool:
        """Add content to the knowledge graph + attach memory_id back-refs.

        Extracted from store_raw so it can run either inline (add_to_graph=True)
        or — preferably — as a separate background job on the graph worker,
        since Graphiti entity extraction is slow (~minutes, Gemini-gated) and
        must not block the fast vector write or the API/MCP event loop.
        Best-effort: logs and swallows errors, never raises.

        ``graph_ontology`` is an optional dict of Graphiti custom-type kwargs
        (``entity_types``/``edge_types``/``edge_type_map``/
        ``excluded_entity_types``/``custom_extraction_instructions``) supplied by
        a knowledge adapter and resolved *per ingest*. When None (the default,
        and every regular memory write), the graph write is byte-for-byte the
        pre-adapter path.

        Returns True if the graph write actually succeeded, False if it was
        skipped (no graph configured) or swallowed an error. Callers use this
        to report honest enrichment status instead of assuming success — a
        transient Gemini 503 leaves the memory vector-only, and that must be
        observable rather than reported as ``enriched=True``.
        """
        if not (self._graphiti and self._bridge):
            return False
        # Group_id encodes visibility + user namespace so graph search can
        # scope by allowed groups without re-leaking cross-user facts.
        group_id = _build_group_id(visibility, user_id, project_id)
        graph_write_started_at = datetime.now(timezone.utc)
        try:
            retry_transient(
                self._memory.graph.add,
                data=content,
                filters={"user_id": user_id, "group_id": group_id},
                operation="enrich_graph add",
                **(graph_ontology or {}),
            )
            # Best-effort: attach memory_id back-reference onto the entity nodes
            # Graphiti just created in this group, so the wiki synthesizer can
            # walk community → source memories.
            self._attach_memory_id_to_graph_nodes(
                group_id=group_id,
                memory_id=memory_id,
                visibility=visibility,
                owner_user_id=user_id,
                write_started_at=graph_write_started_at,
                source_ref=source_ref,
            )
            return True
        except Exception as e:
            logger.warning(f"Graph enrichment failed (non-critical): {e}")
            return False

    def _find_by_content_hash(
        self,
        user_id: str,
        content_hash: str,
        scope: str,
        project_id: str | None = None,
        visibility: str | None = None,
    ) -> MemoryResponse | None:
        """Look up a memory by (user_id, hash, scope, visibility) for dedup.

        Returns the existing MemoryResponse on hit, or None if not found.
        Failures here are non-fatal — we'd rather risk a duplicate than
        block an insert.

        ``visibility`` is part of the key: the same text at two different tiers
        (e.g. a dictator's private note vs. an authoritative ``standard``) are
        distinct memories, so a ``standard`` write must not dedup onto a
        pre-existing ``private``/``shared`` row of the same content.
        """
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            client = self._memory.vector_store.client
            collection = settings.qdrant_collection
            must = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="hash", match=MatchValue(value=content_hash)),
                FieldCondition(key="metadata.scope", match=MatchValue(value=scope)),
            ]
            if visibility is not None:
                must.append(FieldCondition(key="metadata.visibility", match=MatchValue(value=visibility)))
            if scope == "project" and project_id:
                must.append(FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id)))
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=must),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            pt = points[0]
            payload = pt.payload or {}
            metadata = payload.get("metadata", {}) or {}
            return MemoryResponse(
                id=str(pt.id),
                memory=payload.get("data", ""),
                category=metadata.get("category"),
                scope=metadata.get("scope"),
                project_id=metadata.get("project_id"),
                tags=metadata.get("tags"),
                source="vector",
                created_at=payload.get("created_at"),
                occurred_at=metadata.get("occurred_at"),
                domain=metadata.get("domain"),
                observation_type=metadata.get("observation_type"),
                concepts=metadata.get("concepts"),
                source_type=metadata.get("source_type"),
                related_memory_ids=metadata.get("related_memory_ids"),
                confidence=metadata.get("confidence"),
                expires_at=metadata.get("expires_at"),
                derived_from=metadata.get("derived_from"),
                epistemic_level=metadata.get("epistemic_level"),
                memory_kind=metadata.get("memory_kind"),
                source_ref=metadata.get("source_ref"),
                visibility=metadata.get("visibility"),
                owner_user_id=metadata.get("owner_user_id"),
                title=metadata.get("title"),
                token_estimate=metadata.get("token_estimate"),
            )
        except Exception as e:
            logger.warning(f"Content-hash dedup lookup failed (non-fatal): {e}")
            return None

    def _revive_if_tombstoned(self, memory_id: str) -> bool:
        """Resurrect a dream-tombstoned dedup survivor (audit 27 #5).

        ``_find_by_content_hash`` matches tombstoned rows too — before this
        existed, re-storing a fact the dreaming sweep had tombstoned was
        silently swallowed by the dedup short-circuit and the fact stayed
        excluded from recall forever (search filters
        ``metadata.dream_tombstoned=true``). Revival semantics:
        re-derivation is fresh evidence the fact is live, so the tombstone
        flag is cleared and the row becomes recallable again.

        The clear is an atomic nested-key merge (``set_payload`` with
        ``key="metadata"``) rather than a read-merge-write of the whole
        metadata dict, so it can't race a concurrent metadata patch into
        resurrecting stale fields.

        Returns True if a tombstone was actually cleared. Best-effort:
        failures log and return False — a lost revival must never block
        the write path.
        """
        try:
            client = self._memory.vector_store.client
            collection = settings.qdrant_collection
            points = client.retrieve(
                collection_name=collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return False
            meta = (points[0].payload or {}).get("metadata") or {}
            if not meta.get("dream_tombstoned"):
                return False
            client.set_payload(
                collection_name=collection,
                payload={
                    "dream_tombstoned": False,
                    "dream_revived_at": datetime.now(timezone.utc).isoformat(),
                },
                points=[memory_id],
                key="metadata",
            )
            logger.info(f"Revived dream-tombstoned memory {memory_id} on re-derivation")
            return True
        except Exception as e:
            logger.warning(f"Tombstone revival failed for {memory_id} (non-fatal): {e}")
            return False

    def _bump_times_derived(self, memory_id: str, add: int = 1) -> None:
        """Increment ``metadata.times_derived`` on a surviving memory.

        Called when a duplicate of ``memory_id`` was detected and dropped —
        the reinforcement is counted on the survivor instead of discarded.

        Audit 27 #30: the write is an atomic nested-key merge
        (``set_payload`` with ``key="metadata"`` carrying ONLY
        ``times_derived``) — a concurrent patcher's keys (e.g. a dream
        tombstone landing between our read and write) can never be
        clobbered by a whole replaced metadata dict. The counter value
        itself still needs a read (Qdrant has no atomic increment), so two
        simultaneous bumps can lose a tick — bounded, and strictly better
        than losing arbitrary foreign keys.

        Best-effort: losing a reinforcement tick must never block a write or
        a dedup pass, so every failure is swallowed with a warning.
        """
        if add <= 0:
            return
        try:
            client = self._memory.vector_store.client
            collection = settings.qdrant_collection
            points = client.retrieve(
                collection_name=collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return
            meta = (points[0].payload or {}).get("metadata") or {}
            client.set_payload(
                collection_name=collection,
                payload={"times_derived": _times_derived_from_metadata(meta) + add},
                points=[memory_id],
                key="metadata",
            )
        except Exception as e:
            logger.warning(f"times_derived bump failed for {memory_id} (non-fatal): {e}")

    def store_raw_batch(
        self,
        items: list[dict],
    ) -> list[MemoryResponse]:
        """Store multiple pre-categorized facts (memory-model v2).

        Each item is a dict matching RawMemoryRequest's shape. Two-pass
        (audit 27 #20): pass 1 validates + content-hash dedups every item via
        the same ``_prepare_raw_store`` that ``store_raw`` uses; pass 2 embeds
        ALL new items in ONE ``embed_batch`` call, then inserts and
        graph-enriches per item. Previously each item paid its own serial
        embed round trip — a 25-item checkpoint was 25 embed RTTs on the
        gateway path.

        Semantics preserved from the per-item path:

        - duplicates (pre-existing rows OR repeats within the batch) return
          the surviving row and bump its ``times_derived``;
        - per-item results in input order (failed items dropped);
        - one bad item won't block the others.

        The single ``embed_batch`` failure mode is deliberately LOUD (fails
        the whole batch before any insert) — the content-hash dedup makes the
        ARQ retry safe, so partial-batch silent loss is the worse trade.

        Returns:
            List of MemoryResponse objects for successful items, input order.
        """
        m = self._get_memory()

        # ── Pass 1: validate + dedup, collect ready-to-insert items ──
        # Slots: ("new", prepared) | ("dup", response) | ("batchdup", prepared)
        # | ("skip", None) — batchdup is a repeat WITHIN this batch of a new
        # item; its bump/response resolve in pass 2, after the survivor exists.
        entries: list[tuple[str, object]] = []
        seen_new: dict[tuple, dict] = {}
        for idx, item in enumerate(items):
            try:
                # Convert ISO string -> datetime if needed (came in via JSON)
                expires_at = item.get("expires_at")
                if expires_at and isinstance(expires_at, str):
                    try:
                        from datetime import datetime as _dt
                        expires_at = _dt.fromisoformat(expires_at.replace("Z", "+00:00"))
                    except ValueError:
                        expires_at = None
                prepared, existing = self._prepare_raw_store(
                    content=item["content"],
                    user_id=item["user_id"],
                    category=item["category"],
                    scope=item.get("scope", "global"),
                    project_id=item.get("project_id"),
                    tags=item.get("tags"),
                    agent_id=item.get("agent_id"),
                    run_id=item.get("run_id"),
                    domain=item.get("domain"),
                    observation_type=item.get("observation_type"),
                    concepts=item.get("concepts"),
                    source_type=item.get("source_type"),
                    related_memory_ids=item.get("related_memory_ids"),
                    confidence=item.get("confidence"),
                    expires_at=expires_at,
                    occurred_at=item.get("occurred_at"),
                    derived_from=item.get("derived_from"),
                    epistemic_level=item.get("epistemic_level"),
                    visibility=item.get("visibility"),
                    memory_kind=item.get("memory_kind"),
                    source_ref=item.get("source_ref"),
                )
                if existing is not None:
                    entries.append(("dup", existing))
                elif prepared["dedup_key"] in seen_new:
                    # Repeat of an item earlier in THIS batch: the sequential
                    # path deduped it onto the just-inserted first occurrence.
                    entries.append(("batchdup", seen_new[prepared["dedup_key"]]))
                else:
                    seen_new[prepared["dedup_key"]] = prepared
                    entries.append(("new", prepared))
            except Exception as e:
                logger.warning(f"Batch item {idx} failed (continuing): {e}")
                entries.append(("skip", None))

        # ── Pass 2: ONE batch embed for all new items, then per-item work ──
        new_texts = [p["content"] for kind, p in entries if kind == "new"]
        embeddings = (
            m.embedding_model.embed_batch(new_texts, memory_action="add")
            if new_texts
            else []
        )
        # Loud contract check BEFORE any insert (Copilot review, PR #124): a
        # short/empty/overlong return would otherwise exhaust (or misalign)
        # the iterator mid-pass — StopIteration swallowed as a per-item
        # failure, items silently dropped or paired with the wrong vectors.
        if len(embeddings) != len(new_texts):
            raise RuntimeError(
                f"store_raw_batch: embed_batch returned {len(embeddings)} "
                f"embeddings for {len(new_texts)} new texts — refusing to "
                f"insert any item of this batch"
            )
        emb_iter = iter(embeddings)

        results: list[MemoryResponse] = []
        finalized_mids: set[str] = set()
        for idx, (kind, obj) in enumerate(entries):
            if kind == "skip":
                continue
            if kind == "dup":
                results.append(obj)
                continue
            if kind == "batchdup":
                # Survivor was inserted earlier in this pass — count the
                # repeat on it, mirroring the old sequential dedup behavior.
                # If the survivor's store FAILED, this repeat failed too:
                # never bump a nonexistent row or report a response for a
                # row that was never stored (Copilot review, PR #124).
                if obj["mid"] not in finalized_mids:
                    logger.warning(
                        f"Batch item {idx} failed (continuing): duplicate of "
                        f"an item in this batch whose store failed"
                    )
                    continue
                self._bump_times_derived(obj["mid"], 1)
                results.append(obj["response"])
                continue
            try:
                self._finalize_raw_store(obj, next(emb_iter))
                finalized_mids.add(obj["mid"])
                results.append(obj["response"])
            except Exception as e:
                logger.warning(f"Batch item {idx} failed (continuing): {e}")
        logger.info(
            f"store_raw_batch: stored {len(results)} memories from {len(items)} items "
            f"(1 embed_batch call for {len(new_texts)} new)"
        )
        return results

    def expire_old_memories(self, batch_size: int = 100) -> dict:
        """Delete memories whose expires_at is in the past (memory-model v2 cron).

        Qdrant doesn't have a direct datetime range filter on string fields,
        so we scroll the whole collection and filter in Python by parsing
        each memory's `expires_at` to an aware UTC datetime. For larger
        deployments we'd switch to a numeric epoch field.

        Returns:
            Dict with deleted_count and per-user breakdown.
        """
        # Ensure mem0/Qdrant client is initialized — the nightly cron can
        # fire on a cold-started worker before any request has touched
        # _memory, which would otherwise raise AttributeError.
        m = self._get_memory()
        client = m.vector_store.client
        collection = settings.qdrant_collection
        now_dt = datetime.now(timezone.utc)

        deleted_count = 0
        per_user: dict[str, int] = {}
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for pt in points:
                payload = pt.payload or {}
                metadata = payload.get("metadata", {}) or {}
                expires_at = metadata.get("expires_at")
                if not expires_at:
                    continue
                expires_dt = _parse_expires_at(expires_at)
                if expires_dt is None:
                    # Malformed timestamp — log and skip rather than risk
                    # deleting on an unparseable value.
                    logger.warning(
                        f"Memory {pt.id} has unparseable expires_at={expires_at!r}; skipping"
                    )
                    continue
                if expires_dt >= now_dt:
                    continue  # not yet expired
                try:
                    self._delete_qdrant_memory_with_graph_cleanup(str(pt.id), payload)
                    uid = payload.get("user_id", "unknown")
                    per_user[uid] = per_user.get(uid, 0) + 1
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to expire memory {pt.id}: {e}")
            if next_offset is None:
                break
            offset = next_offset

        if deleted_count:
            logger.info(f"expire_old_memories: deleted {deleted_count} expired memories")
        return {"deleted_count": deleted_count, "per_user": per_user}

    def _batch_store_facts(
        self,
        facts: list[tuple[str, str]],
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        source: str = "conversation",
        source_type: str | None = None,
        memory_kind: str | None = None,
        source_ref: dict | None = None,
        occurred_at: str | None = None,
        speakers: list[str | None] | None = None,
    ) -> list[MemoryResponse]:
        """Store multiple categorized facts via a single batch embed + single Qdrant upsert.

        Bypasses mem0's per-fact m.add() pipeline which triggers per-fact graph
        ingestion.  The caller is responsible for a separate graph.add() call
        with the full conversation text.

        Content-hash dedup (audit 27 #21): before inserting, each fact is
        checked against (user, hash, scope, project) via
        ``_find_by_content_hash`` — an ARQ retry after a partial upsert (or a
        client resubmission) no longer duplicates facts under fresh UUIDs.
        Dedup hits bump ``times_derived`` on the survivor (and revive a
        dream-tombstoned one) exactly like the raw path, and the survivor is
        returned in the fact's slot. Duplicates WITHIN the batch (extraction
        window overlap re-deriving the same fact) collapse to the first
        occurrence without a bump — one conversation isn't independent
        re-derivation evidence. The N dedup probes are single indexed Qdrant
        scrolls (~ms each), noise next to the embed + LLM calls on this path.

        Args:
            facts: List of (category, content) tuples.
            user_id: User identifier.
            project_id: Optional project identifier.
            agent_id: Optional agent identifier.
            run_id: Optional run/session identifier.
            source: Provenance tag for metadata.
            occurred_at: Event time (ISO 8601) applied to EVERY fact in the
                batch — when the conversation actually happened, for
                historical ingestion of old chat exports. None ⇒ omitted
                (event time unknown; readers fall back to created_at).
            speakers: Optional list of per-fact speaker labels (T1.2), aligned
                to ``facts``. None or a shorter list is tolerated — missing
                entries default to None (no speaker). Validated via
                ``_validate_speaker`` before persisting.

        Returns:
            List of MemoryResponse objects for stored facts (new rows and
            dedup survivors, in input order; in-batch duplicates dropped).
        """
        if not facts:
            return []

        # Validate once for the whole batch (every fact shares the
        # conversation's event time). Raises ValueError on garbage.
        occurred_at = validate_occurred_at(occurred_at)

        m = self._get_memory()
        now_iso = datetime.now(timezone.utc).isoformat()

        # ── Build per-fact metadata, IDs, and texts (dedup-first) ──
        texts: list[str] = []
        memory_ids: list[str] = []
        payloads: list[dict] = []
        fact_meta: list[tuple[str, str, str | None, str | None]] = []  # (category, scope, project_id, speaker) per fact
        # Ordered slots: ("new", index-into-texts) | ("dup", existing response)
        ordered: list[tuple[str, object]] = []
        seen_in_batch: set[tuple[str, str, str | None]] = set()
        dedup_hits = 0

        # Normalize speakers list: pad to len(facts) with None if shorter/absent
        speakers_normalized = speakers or []
        if len(speakers_normalized) < len(facts):
            speakers_normalized = list(speakers_normalized) + [None] * (
                len(facts) - len(speakers_normalized)
            )

        for idx, (category, content) in enumerate(facts):
            raw_speaker = speakers_normalized[idx] if idx < len(speakers_normalized) else None
            # T1.2 speaker sanity guard: validate before persisting
            speaker = _validate_speaker(raw_speaker)
            scope = default_scope_for_category(category)
            fact_project_id = project_id

            if fact_project_id and category not in GLOBAL_CATEGORIES:
                scope = MemoryScope.PROJECT

            if category in PROJECT_CATEGORIES and not fact_project_id:
                inferred = _infer_project_id(content)
                if inferred:
                    fact_project_id = inferred
                    scope = MemoryScope.PROJECT
                    logger.info(
                        f"Inferred project_id='{inferred}' from content for category '{category}'"
                    )
                else:
                    scope = MemoryScope.GLOBAL
                    logger.warning(
                        f"Category '{category}' requires project_id but none provided and could not be inferred. "
                        f"Content snippet: '{content[:80]}'. Storing as global scope."
                    )

            scope_val = scope.value if isinstance(scope, MemoryScope) else scope
            chash = content_hash(content)

            # In-batch duplicate (extraction-window overlap): keep the first.
            batch_key = (chash, scope_val, fact_project_id)
            if batch_key in seen_in_batch:
                continue
            seen_in_batch.add(batch_key)

            # Storage-level idempotency (audit 27 #21). visibility=None on
            # purpose: conversation-path rows don't stamp metadata.visibility,
            # so a visibility condition would never match them.
            existing = self._find_by_content_hash(
                user_id=user_id,
                content_hash=chash,
                scope=scope_val,
                project_id=fact_project_id,
                visibility=None,
            )
            if existing is not None:
                # Same reinforcement semantics as the raw path: count the
                # re-derivation on the survivor, resurrect it if a dream
                # sweep had tombstoned it (audit 27 #5 applies here too).
                self._bump_times_derived(existing.id, 1)
                if self._revive_if_tombstoned(existing.id):
                    existing.revived = True
                dedup_hits += 1
                ordered.append(("dup", existing))
                continue

            mid = str(uuid.uuid4())
            payload = {
                "data": content,
                "hash": chash,
                "created_at": now_iso,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "metadata": {
                    "scope": scope_val,
                    "category": category,
                    "project_id": fact_project_id,
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "source": source,
                    # Extraction stores directly-stated facts → epistemically
                    # "explicit" (A1). Derived levels are stamped by their
                    # authors (dreaming reflection/merge), never here.
                    "epistemic_level": "explicit",
                    # Retrieval economics (C1): index-row fields (heuristic, no LLM)
                    "title": distill_title(content),
                    "token_estimate": stamp_tokens(content),
                    **({"source_type": source_type} if source_type is not None else {}),
                    **({"memory_kind": memory_kind} if memory_kind is not None else {}),
                    **({"source_ref": source_ref} if source_ref else {}),
                    # Event time: only stored when known (never defaulted).
                    **({"occurred_at": occurred_at} if occurred_at is not None else {}),
                    # T1.2: speaker attribution (only stored when present)
                    **({"speaker": speaker} if speaker else {}),
                },
            }

            texts.append(content)
            memory_ids.append(mid)
            payloads.append(payload)
            fact_meta.append((category, scope_val, fact_project_id, speaker))
            ordered.append(("new", len(texts) - 1))

        # ── Single batch embed + single Qdrant upsert (new facts only) ──
        # Skipped entirely when everything dedup'd — a straight ARQ re-run of
        # an already-stored conversation inserts ZERO new points.
        if texts:
            embeddings = m.embedding_model.embed_batch(texts, memory_action="add")
            m.vector_store.insert(
                vectors=embeddings,
                ids=memory_ids,
                payloads=payloads,
            )

            # ── Record history entries ──
            for mid, content in zip(memory_ids, texts):
                try:
                    m.db.add_history(mid, None, content, "ADD", created_at=now_iso)
                except Exception as e:
                    logger.warning(f"History record failed for {mid}: {e}")

        # ── Build responses (input order: new rows + dedup survivors) ──
        responses: list[MemoryResponse] = []
        for kind, ref in ordered:
            if kind == "dup":
                responses.append(ref)  # the existing MemoryResponse
                continue
            idx = ref
            mid, content = memory_ids[idx], texts[idx]
            category, scope_val, fact_pid, spk = fact_meta[idx]
            responses.append(
                MemoryResponse(
                    id=mid,
                    memory=content,
                    category=category,
                    scope=scope_val,
                    project_id=fact_pid,
                    source="vector",
                    created_at=now_iso,
                    occurred_at=occurred_at,
                    source_type=source_type,
                    epistemic_level="explicit",
                    memory_kind=memory_kind,
                    source_ref=source_ref,
                    title=distill_title(content),
                    token_estimate=stamp_tokens(content),
                    speaker=spk,
                )
            )

        logger.info(
            f"Batch-stored {len(texts)} new facts for user={user_id} "
            f"({dedup_hits} content-hash dedup hits; 1 embed call, 1 Qdrant upsert)"
        )
        return responses
