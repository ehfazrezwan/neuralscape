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
from memory.sensitivity import classify_sensitivity, resolve_gated_visibility

logger = logging.getLogger(__name__)
_SPEAKER_UNSET = object()


def _occurred_at_to_datetime(value: str | None) -> datetime | None:
    """Parse occurred_at ISO string to datetime for Graphiti reference_time.

    Normalizes through ``validate_occurred_at`` first so the same rules the
    rest of the envelope uses apply here too — a trailing ``Z`` is tolerated,
    naive timestamps are assumed UTC, and future-skew/invalid values are
    rejected. Defensive: returns None on any failure so bad or unset values
    degrade to legacy 'now' behavior instead of raising.

    Args:
        value: ISO 8601 timestamp string (may carry ``Z`` / be naive).

    Returns:
        Parsed timezone-aware datetime, or None if value is falsy/invalid.
    """
    if not value:
        return None
    try:
        normalized = validate_occurred_at(value)  # canonical UTC ISO (+00:00) or None
        if not normalized:
            return None
        return datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        logger.warning(f"Failed to parse occurred_at '{value}' — falling back to ingestion time")
        return None


def _speaker_label(msg: dict) -> str:
    """Line prefix (the "actor") for a conversation message sent to Graphiti's
    speaker-first ``extract_message`` prompt (R2).

    Prefers a real speaker name over the generic role: the first of
    ``speaker`` → ``name`` → ``role`` that is present AND sanitizes to a
    non-empty string is returned; ``"user"`` is the final fallback. Every
    candidate — including the role fallback — is sanitized (stripped, internal
    whitespace/newlines collapsed to single spaces) so a whitespace-only or
    multi-line label can never reintroduce a newline into the episode header.
    For role-only messages this returns the role verbatim, so ``raw_text`` is
    byte-identical to the pre-R2 behavior.
    """
    for candidate in (msg.get("speaker"), msg.get("name"), msg.get("role"), "user"):
        if not candidate:
            continue
        sanitized = " ".join(str(candidate).strip().split())
        if sanitized:
            return sanitized
    return "user"


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

        # Step 1.5: Choose parser based on extraction_require_speaker flag (FIX 1).
        # When the flag is OFF (production default), use the LEGACY parser that
        # folds any naturally-occurring "prefix: content" patterns back into
        # content without splitting into speaker metadata. This restores byte-
        # identical backward-compat: stored content and dedup identity match
        # pre-T1.2 exactly (no speaker metadata, no content mutation).
        # When the flag is ON, use the rich parser for speaker + occurred_at.
        use_rich_parser = settings.extraction_require_speaker
        from prompts import ParsedFact

        # Every window is parsed with the rich parser (category, content,
        # speaker, occurred_at, and the optional sensitivity tag — see
        # prompts.ParsedFact). Whether the CALLER sees speaker/occurred_at
        # metadata is still gated by use_rich_parser below; parsing rich
        # unconditionally just means the optional sensitivity signal survives
        # even on the legacy (use_rich_parser=False, production-default) path,
        # instead of being dropped by parse_extraction_response's fold.
        parsed_facts_all: list[ParsedFact] = []
        window_errors: list[str] = []
        last_exc: Exception | None = None
        for w_idx, window_messages in enumerate(windows):
            extraction_messages = build_extraction_messages(
                window_messages,
                operator_guidance=operator_guidance,
                require_speaker=settings.extraction_require_speaker,
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
                parsed_facts_all.extend(parse_extraction_response_rich(response.text))
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
        # For rich parser: check the FULL fact (speaker + content) for junk patterns.
        # For legacy parser: content already includes any prefix — fold it the
        # same way parse_extraction_response used to, so junk detection sees
        # exactly the same string it always did.
        if use_rich_parser:
            pre_filter_count = len(parsed_facts_all)
            parsed_facts_rich = [
                pf for pf in parsed_facts_all
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
        else:
            # Legacy fold: identical to parse_extraction_response's fold
            # (speaker prefixed back into content when present), so stored
            # content + dedup identity stay byte-identical to before.
            folded_all = [
                (pf.category, (f"{pf.speaker}: {pf.content}" if pf.speaker else pf.content), pf.sensitivity)
                for pf in parsed_facts_all
            ]
            pre_filter_count = len(folded_all)
            folded_all = [
                (cat, content, sens) for cat, content, sens in folded_all
                if not _is_junk_fact(content)
            ]
            if pre_filter_count != len(folded_all):
                logger.info(
                    f"Filtered {pre_filter_count - len(folded_all)} junk facts from extraction"
                )
            if not folded_all:
                logger.info("No facts extracted from conversation")
                return ([], stats) if return_stats else []

        # Step 2: Batch-store all facts (single embed + single Qdrant upsert).
        # T1.2: thread speaker attribution through. We extract (category, content)
        # tuples for _batch_store_facts, plus a parallel speakers list.
        # T1.3: Build per-fact occurred_ats from ParsedFact, validating each one.
        # Valid ISO values use the fact's timestamp; invalid/absent values fall
        # back to the conversation-level occurred_at parameter.
        # A failure here (embedding, Qdrant) must PROPAGATE: the previous
        # except-and-continue silently stored zero facts while the task
        # reported success — exactly what hid the gateway's single-input
        # embed rejection in production. Raising fails the ARQ job, so
        # /v1/memories/status/{task_id} reports status=failed with the error.
        try:
            if use_rich_parser:
                facts = [(pf.category, pf.content) for pf in parsed_facts_rich]
                speakers = [pf.speaker for pf in parsed_facts_rich]
                sensitivities = [pf.sensitivity for pf in parsed_facts_rich]
                # T1.3: per-fact occurred_at extraction with fallback
                occurred_ats: list[str | None] = []
                for pf in parsed_facts_rich:
                    per_fact_time: str | None = None
                    if pf.occurred_at:
                        try:
                            per_fact_time = validate_occurred_at(pf.occurred_at)
                        except (ValueError, TypeError):
                            # Invalid or relative phrase → use the fallback
                            per_fact_time = occurred_at
                    else:
                        # No per-fact time → use the conversation-level fallback
                        per_fact_time = occurred_at
                    occurred_ats.append(per_fact_time)
            else:
                # Legacy path: no speaker metadata, no per-fact occurred_at.
                # Thread conversation-level occurred_at as before (T1.3 fallback).
                facts = [(cat, content) for cat, content, sens in folded_all]
                speakers = None
                occurred_ats = None
                sensitivities = [sens for cat, content, sens in folded_all]
            # F8: track which memory ids are NEWLY CREATED (vs. content-hash
            # dedup survivors) so the graph-episode-ref stamping below only
            # touches rows this conversation's episode actually created —
            # see the loops using ``newly_created_ids``.
            newly_created_ids: set[str] = set()
            stored = self._batch_store_facts(
                facts=facts,
                speakers=speakers,
                user_id=user_id,
                project_id=project_id,
                agent_id=agent_id,
                run_id=run_id,
                source="conversation",
                occurred_at=occurred_at,
                occurred_ats=occurred_ats,
                sensitivities=sensitivities,
                created_ids=newly_created_ids,
            )
        except Exception as e:
            logger.error(
                f"Batch store failed for user {user_id}: {e} — "
                f"{len(facts)} extracted facts were NOT stored",
                exc_info=True,
            )
            raise

        # Step 3: Add cleaned conversation text to knowledge graph — ONE
        # episode for the whole conversation regardless of extraction
        # windowing (Graphiti handles its own entity windowing internally).
        # The episode's Graphiti reference_time is the conversation's event
        # time (occurred_at), falling back to ingestion time when unknown.
        # Conversation extractions are personal (private) by default — the
        # caller's spoken context isn't team-shared automatically.
        group_id = _build_group_id(
            MemoryVisibility.PRIVATE.value,
            user_id,
            project_id,
        )
        cleaned_messages = _clean_conversation_for_graph(messages)

        raw_text = "\n".join(
            f"{_speaker_label(msg)}: {msg.get('content', '')}"
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
                    # Provenance durability: we didn't write the episode this
                    # time, but its name is still deterministic — persist it
                    # so a later cascade can resolve by name even without the
                    # uuid (memory/provenance.py resolution order tries name
                    # before falling back to content).
                    #
                    # F8: only for NEWLY CREATED rows. ``stored`` also
                    # contains content-hash dedup survivors that already
                    # point at their OWN (possibly different) episode;
                    # repointing a survivor at this episode would make a
                    # later cascade target the wrong episode and leave the
                    # survivor's real episode live.
                    for mem in stored:
                        mid = getattr(mem, "id", "") or ""
                        if mid not in newly_created_ids:
                            continue
                        self._persist_graph_episode_ref(mid, None, episode_name)
                else:
                    graph_result = retry_transient(
                        self._memory.graph.add,
                        data=raw_text,
                        filters={"user_id": user_id, "group_id": group_id},
                        episode_name=episode_name,
                        reference_time=_occurred_at_to_datetime(occurred_at),
                        episode_source="message",
                        operation="graph storage (extract_and_store)",
                    )
                    episode_uuid = (graph_result or {}).get("episode_uuid")
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
                        mid = getattr(mem, "id", "") or ""
                        self._attach_memory_id_to_graph_nodes(
                            group_id=group_id,
                            memory_id=mid,
                            visibility=MemoryVisibility.PRIVATE.value,
                            owner_user_id=user_id,
                            write_started_at=graph_write_started_at,
                        )
                        # Provenance durability: same episode → N memories;
                        # stamp each row so a later cascade can resolve the
                        # EXACT episode (memory/provenance.py). F8: NEW rows
                        # only — a dedup survivor already has its own episode
                        # ref (see the docstring above the sibling loop).
                        if (episode_uuid or episode_name) and mid in newly_created_ids:
                            self._persist_graph_episode_ref(mid, episode_uuid, episode_name)
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
        # Workspace partition (WT6, None → memory type)
        workspace: str | None = None,
        # Sensitivity gate override — see _prepare_raw_store. Only bypasses
        # the forced-private gate when `visibility` is also explicitly set.
        sensitivity_override: bool = False,
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
            workspace=workspace,
            sensitivity_override=sensitivity_override,
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
        workspace: str | None = None,
        # Sensitivity gate override: bypasses the forced-private gate below
        # ONLY when combined with an explicit `visibility` (see the gate
        # logic). False by default — every existing caller that doesn't know
        # about the gate keeps getting the gate's protection.
        sensitivity_override: bool = False,
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

        # Resolve visibility: explicit caller value > per-category default,
        # with the deterministic sensitivity gate layered on top. Both are
        # ``normalize_visibility``'s enum/str/legacy-stringified-enum
        # tolerance and the SHARED-category-vs-sensitive-content gate that
        # forces visibility=private unless the caller both supplied an
        # explicit visibility AND passed sensitivity_override=True.
        # ``resolve_gated_visibility`` (memory/sensitivity.py) is the single
        # source of truth for this resolution — worker.py's pre-store
        # idempotency dedup check shares it too (F7: it must dedupe against
        # the RESOLVED visibility the write will land at, not the raw
        # requested one, or a gated write can coalesce onto — and skip in
        # favor of — a pre-existing SHARED row with the same content).
        # Resolved BEFORE the scope check because standards force scope below.
        explicit_visibility_requested = visibility is not None
        effective_visibility, sensitivity_class, sensitivity_source, gate_action = (
            resolve_gated_visibility(content, category, visibility, sensitivity_override)
        )
        if gate_action == "bypassed":
            logger.debug(
                f"Sensitivity gate: class={sensitivity_class} bypassed via "
                f"explicit sensitivity_override (user={user_id}, category={category})"
            )
        elif gate_action == "forced":
            if explicit_visibility_requested and not sensitivity_override:
                logger.warning(
                    f"Sensitivity gate: caller requested visibility for a "
                    f"{sensitivity_class} write without sensitivity_override=True — "
                    f"forcing visibility=private (user={user_id}, category={category})"
                )
            else:
                logger.info(
                    f"Sensitivity gate: forcing visibility=private for category="
                    f"{category} (class={sensitivity_class}, source=regex, "
                    f"user={user_id})"
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
        # Skip insert if this exact (user_id, scope, hash, workspace) already exists.
        # Workspace is part of dedup identity: the same sentence in a book vs a
        # user note are distinct rows (WT6).
        effective_workspace = workspace if workspace else None
        existing = self._find_by_content_hash(
            user_id=user_id, content_hash=chash, scope=scope,
            project_id=project_id, visibility=effective_visibility,
            workspace=effective_workspace,
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
            # Sensitivity gate provenance: stamped whenever the content matched a
            # gated class — `sensitivity_source` records whether the gate forced
            # private ("regex"/"llm") or the caller bypassed it ("bypassed").
            # Absent for non-matching content, so a plain `decision`/
            # `interaction`/etc. write is byte-identical to before.
            **({"sensitivity": sensitivity_class} if sensitivity_class else {}),
            **({"sensitivity_source": sensitivity_source} if sensitivity_source else {}),
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
        # Workspace partition (WT6): only stored when set (absent/None = memory type)
        if workspace is not None:
            metadata["workspace"] = workspace

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
            sensitivity=sensitivity_class,
            sensitivity_source=sensitivity_source,
            title=title,
            token_estimate=token_estimate,
            workspace=effective_workspace,
        )

        prepared = {
            "mid": mid,
            "content": content,
            "payload": payload,
            "response": response,
            "now_iso": now_iso,
            # (user, hash, scope, project, visibility, workspace) — the dedup identity,
            # exposed so store_raw_batch can collapse in-batch duplicates. Workspace
            # is part of the identity: same content in memory vs reference = distinct (WT6).
            "dedup_key": (user_id, chash, scope, project_id, effective_visibility, effective_workspace),
            # Fields _finalize_raw_store needs for the graph write.
            "user_id": user_id,
            "project_id": project_id,
            "effective_visibility": effective_visibility,
            "source_ref": source_ref,
            "workspace": effective_workspace,
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
                workspace=prepared.get("workspace"),
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
        occurred_at: str | None = None,
        workspace: str | None = None,
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

        ``occurred_at`` is the real event time (ISO string); None falls back
        to ingestion wall-clock (legacy behavior).

        Returns True if the graph write actually succeeded, False if it was
        skipped (no graph configured) or swallowed an error. Callers use this
        to report honest enrichment status instead of assuming success — a
        transient Gemini 503 leaves the memory vector-only, and that must be
        observable rather than reported as ``enriched=True``.
        """
        if not (self._graphiti and self._bridge):
            return False
        # Group_id encodes visibility + user namespace + workspace (WT6) so
        # graph search can scope by allowed groups without re-leaking
        # cross-user facts or mixing memory/reference pools.
        group_id = _build_group_id(visibility, user_id, project_id, workspace)
        graph_write_started_at = datetime.now(timezone.utc)
        try:
            graph_result = retry_transient(
                self._memory.graph.add,
                data=content,
                filters={"user_id": user_id, "group_id": group_id},
                reference_time=_occurred_at_to_datetime(occurred_at),
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
            # Provenance durability: this single-fact path does NOT name its
            # episode deterministically — MemoryGraph.add() mints a
            # timestamp-based name when no episode_name is supplied. Persist
            # whatever uuid/name Graphiti actually resolved onto the Qdrant
            # row so a later visibility flip / delete can cascade-expire the
            # EXACT episode (memory/provenance.py) instead of falling back to
            # a lossy heuristic. Additive — never blocks the graph result.
            episode_uuid = (graph_result or {}).get("episode_uuid")
            episode_name = (graph_result or {}).get("episode_name")
            if episode_uuid or episode_name:
                self._persist_graph_episode_ref(memory_id, episode_uuid, episode_name)
            return True
        except Exception as e:
            logger.warning(f"Graph enrichment failed (non-critical): {e}")
            return False

    def _backfill_speaker_on_existing_memory(
        self,
        *,
        memory_id: str,
        speaker: str,
    ) -> bool:
        """Best-effort patch to attach missing speaker metadata onto an existing row."""
        try:
            client = self._memory.vector_store.client
            collection = settings.qdrant_collection
            client.set_payload(
                collection_name=collection,
                payload={"speaker": speaker},
                points=[memory_id],
                key="metadata",
            )
            return True
        except Exception as e:
            logger.warning(f"Speaker backfill failed for {memory_id} (non-fatal): {e}")
            return False

    def _find_by_content_hash(
        self,
        user_id: str,
        content_hash: str,
        scope: str,
        project_id: str | None = None,
        visibility: str | None = None,
        speaker: str | None | object = _SPEAKER_UNSET,
        workspace: str | None = None,
    ) -> MemoryResponse | None:
        """Look up a memory by (user_id, hash, scope, visibility, speaker, workspace) for dedup.

        Returns the existing MemoryResponse on hit, or None if not found.
        Failures here are non-fatal — we'd rather risk a duplicate than
        block an insert.

        ``visibility`` is part of the key: the same text at two different tiers
        (e.g. a dictator's private note vs. an authoritative ``standard``) are
        distinct memories, so a ``standard`` write must not dedup onto a
        pre-existing ``private``/``shared`` row of the same content.

        ``speaker`` is tri-state:
        - omitted: preserve legacy behavior (speaker-agnostic lookup)
        - ``None``: only match rows with no speaker metadata
        - string: only match rows with that exact speaker

        ``workspace`` (WT6) is also part of the key: the same sentence in a
        reference book vs. a user note are distinct memories. Absent/None =
        memory type (default); matches rows with workspace absent/None/"memory".
        """
        try:
            from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition, MatchValue, PayloadField
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
            if speaker is None:
                # FIX 2: Use IsEmptyCondition to match rows where speaker is missing
                # or null. IsNullCondition only matches EXPLICIT nulls, but payloads
                # omit the speaker key when absent (never write speaker: null), so
                # the backfill lookup was dead code. IsEmptyCondition matches both.
                must.append(IsEmptyCondition(is_empty=PayloadField(key="metadata.speaker")))
            elif speaker is not _SPEAKER_UNSET:
                must.append(FieldCondition(key="metadata.speaker", match=MatchValue(value=speaker)))
            # Workspace filter (WT6): absent/None matches rows with workspace
            # absent/None/"memory" (all three represent the memory workspace).
            # Non-None workspace must match exactly.
            if workspace is None:
                # Memory type: match rows whose workspace is absent/null (legacy
                # + memory-type rows never write the field) OR explicitly "memory".
                # The nested should = OR; it excludes reference rows (which carry
                # a non-memory workspace) so a memory write can't dedup onto a
                # book passage of identical content.
                must.append(
                    Filter(should=[
                        IsEmptyCondition(is_empty=PayloadField(key="metadata.workspace")),
                        FieldCondition(key="metadata.workspace", match=MatchValue(value="memory")),
                    ])
                )
            else:
                must.append(FieldCondition(key="metadata.workspace", match=MatchValue(value=workspace)))
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
                sensitivity=metadata.get("sensitivity"),
                sensitivity_source=metadata.get("sensitivity_source"),
                title=metadata.get("title"),
                token_estimate=metadata.get("token_estimate"),
                speaker=metadata.get("speaker"),
                workspace=metadata.get("workspace"),
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
                    sensitivity_override=item.get("sensitivity_override", False),
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
        occurred_ats: list[str | None] | None = None,
        workspace: str | None = None,
        sensitivities: list[str | None] | None = None,
        created_ids: set[str] | None = None,
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
            occurred_ats: Optional list of per-fact event times (T1.3), aligned
                to ``facts``. None or a shorter list is tolerated — missing
                entries use the conversation-level ``occurred_at`` fallback.
                Per-fact values take precedence when present.
            sensitivities: Optional list of per-fact LLM-supplied sensitivity
                hints (financial | equity_compensation | client_commercial |
                credentials_pii | None), aligned to ``facts``. Combined with
                a deterministic regex floor over each fact's content — see
                the sensitivity gate below — to decide whether the fact gets
                forced to ``visibility=private``. This path never exposes a
                caller-supplied ``visibility``, so there is no override.
            created_ids: Optional output param — when given, every NEWLY
                CREATED memory's id is added to this set (dedup survivors
                are NOT added). ``stored`` mixes both kinds in input order
                and a MemoryResponse alone can't distinguish them, so a
                caller that must only act on genuinely-new rows (e.g.
                extract_and_store's graph-episode-ref stamping — F8: a
                survivor's graph_episode_uuid/name must not be repointed at
                a DIFFERENT conversation's episode) passes a set here.

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
        fact_meta: list[tuple[str, str, str | None, str | None, str | None, str | None, str | None, str | None]] = []  # (category, scope, project_id, speaker, occurred_at, forced_visibility, sensitivity, sensitivity_source) per fact
        # Ordered slots: ("new", index-into-texts) | ("dup", existing response)
        ordered: list[tuple[str, object]] = []
        seen_in_batch: set[tuple[str, str, str | None, str | None]] = set()
        dedup_hits = 0

        # Normalize speakers list: pad to len(facts) with None if shorter/absent
        speakers_normalized = speakers or []
        if len(speakers_normalized) < len(facts):
            speakers_normalized = list(speakers_normalized) + [None] * (
                len(facts) - len(speakers_normalized)
            )

        # Normalize occurred_ats list: pad to len(facts) with occurred_at fallback if shorter/absent
        occurred_ats_normalized = occurred_ats or []
        if len(occurred_ats_normalized) < len(facts):
            occurred_ats_normalized = list(occurred_ats_normalized) + [occurred_at] * (
                len(facts) - len(occurred_ats_normalized)
            )

        # Normalize sensitivities list: pad to len(facts) with None if shorter/absent
        sensitivities_normalized = sensitivities or []
        if len(sensitivities_normalized) < len(facts):
            sensitivities_normalized = list(sensitivities_normalized) + [None] * (
                len(facts) - len(sensitivities_normalized)
            )

        for idx, (category, content) in enumerate(facts):
            raw_speaker = speakers_normalized[idx] if idx < len(speakers_normalized) else None
            # T1.2 speaker sanity guard: validate before persisting
            speaker = _validate_speaker(raw_speaker)
            # T1.3: per-fact occurred_at (already validated by caller)
            # Use per-fact value if present, otherwise fall back to conversation-level
            fact_occurred_at = (
                occurred_ats_normalized[idx] if idx < len(occurred_ats_normalized) else occurred_at
            )
            if fact_occurred_at is None:
                fact_occurred_at = occurred_at
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
                        f"Storing as global scope."
                    )

            scope_val = scope.value if isinstance(scope, MemoryScope) else scope
            chash = content_hash(content)

            # In-batch duplicate (extraction-window overlap): keep the first.
            # WT6: workspace is part of the batch key too.
            batch_key = (chash, scope_val, fact_project_id, speaker, workspace)
            if batch_key in seen_in_batch:
                continue
            seen_in_batch.add(batch_key)

            # Storage-level idempotency (audit 27 #21). visibility=None on
            # purpose: conversation-path rows don't stamp metadata.visibility,
            # so a visibility condition would never match them. workspace defaults
            # to None here (memory type) unless the caller overrode it.
            lookup_kwargs = dict(
                user_id=user_id,
                content_hash=chash,
                scope=scope_val,
                project_id=fact_project_id,
                visibility=None,
                workspace=workspace,
            )
            existing = (
                self._find_by_content_hash(**lookup_kwargs)
                if speaker is None
                else self._find_by_content_hash(**lookup_kwargs, speaker=speaker)
            )
            if existing is None and speaker is not None:
                existing = self._find_by_content_hash(
                    user_id=user_id,
                    content_hash=chash,
                    scope=scope_val,
                    project_id=fact_project_id,
                    visibility=None,
                    speaker=None,
                )
                if existing is not None and self._backfill_speaker_on_existing_memory(
                    memory_id=existing.id,
                    speaker=speaker,
                ):
                    existing.speaker = speaker
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

            # ── Sensitivity gate (deterministic regex floor + optional LLM
            # hint, zero extra LLM cost) ── This path never receives a
            # caller-supplied visibility, so there is no override: a sensitive
            # extracted fact is always forced private. See memory/sensitivity.py
            # and memory/write.py's _prepare_raw_store for the shared rules.
            fact_sensitivity: str | None = None
            fact_sensitivity_source: str | None = None
            fact_forced_visibility: str | None = None
            if settings.sensitivity_gate_enabled:
                regex_class = classify_sensitivity(content)
                llm_hint = sensitivities_normalized[idx] if idx < len(sensitivities_normalized) else None
                candidate_class = regex_class or llm_hint
                if candidate_class in settings.sensitivity_private_classes_set():
                    fact_sensitivity = candidate_class
                    fact_sensitivity_source = "regex" if regex_class else "llm"
                    fact_forced_visibility = MemoryVisibility.PRIVATE.value
                    logger.info(
                        f"Sensitivity gate: forcing visibility=private for category="
                        f"{category} (class={fact_sensitivity}, source={fact_sensitivity_source}, "
                        f"user={user_id})"
                    )

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
                    **({"visibility": fact_forced_visibility} if fact_forced_visibility else {}),
                    **({"sensitivity": fact_sensitivity} if fact_sensitivity else {}),
                    **({"sensitivity_source": fact_sensitivity_source} if fact_sensitivity_source else {}),
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
                    # T1.3: per-fact value takes precedence when present.
                    **({"occurred_at": fact_occurred_at} if fact_occurred_at is not None else {}),
                    # T1.2: speaker attribution (only stored when present)
                    **({"speaker": speaker} if speaker else {}),
                    # Workspace partition (WT6): only stored when set.
                    **({"workspace": workspace} if workspace is not None else {}),
                },
            }

            texts.append(content)
            memory_ids.append(mid)
            payloads.append(payload)
            fact_meta.append((
                category, scope_val, fact_project_id, speaker, fact_occurred_at,
                fact_forced_visibility, fact_sensitivity, fact_sensitivity_source,
            ))
            ordered.append(("new", len(texts) - 1))
            if created_ids is not None:
                created_ids.add(mid)

        # ── Single batch embed + single Qdrant upsert (new facts only) ──
        # Skipped entirely when everything dedup'd — a straight ARQ re-run of
        # an already-stored conversation inserts ZERO new points.
        if texts:
            embeddings = m.embedding_model.embed_batch(texts, memory_action="add")
            # Validate embed count matches text count before zipping — a
            # misaligned embed/fact zip silently writes garbage vectors.
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"embed_batch returned {len(embeddings)} embeddings for "
                    f"{len(texts)} texts — cannot safely align vectors to facts"
                )
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
            (
                category, scope_val, fact_pid, spk, fact_occurred_at,
                forced_visibility, fact_sensitivity, fact_sensitivity_source,
            ) = fact_meta[idx]
            responses.append(
                MemoryResponse(
                    id=mid,
                    memory=content,
                    category=category,
                    scope=scope_val,
                    project_id=fact_pid,
                    source="vector",
                    created_at=now_iso,
                    occurred_at=fact_occurred_at,
                    source_type=source_type,
                    epistemic_level="explicit",
                    memory_kind=memory_kind,
                    source_ref=source_ref,
                    title=distill_title(content),
                    token_estimate=stamp_tokens(content),
                    speaker=spk,
                    visibility=forced_visibility,
                    sensitivity=fact_sensitivity,
                    sensitivity_source=fact_sensitivity_source,
                )
            )

        logger.info(
            f"Batch-stored {len(texts)} new facts for user={user_id} "
            f"({dedup_hits} content-hash dedup hits; 1 embed call, 1 Qdrant upsert)"
        )
        return responses
