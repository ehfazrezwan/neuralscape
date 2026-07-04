"""Reasoning-tiered question answering over memories (roadmap C3).

``ask_memory`` runs a question-answering pass over the caller's stored
memories instead of returning raw search hits. A single ``reasoning_level``
knob — ``minimal | low | medium | high`` — jointly selects:

- **retrieval breadth**: how many hits each search pass pulls;
- **tool budget / iteration cap**: whether (and how many times) the
  answering LLM may request follow-up searches;
- **mechanical passes**: an update-language search, gated on temporal cues
  in the question or first-pass evidence (so newer facts supersede older
  ones without diluting atemporal asks), and a grep-style exact-keyword
  pass (embeddings under-recall exhaustive enumeration sets);
- **thinking depth**: prompt verbosity;
- **output cap**: maximum answer length.

The answering prompt encodes the Honcho dialectic disciplines:

1. enumeration/counting questions get exact/keyword passes before semantic
   and a dedup table before any count;
2. update-language searches ("changed", "rescheduled", "now") so newer
   facts supersede older ones;
3. contradictions are surfaced BOTH-with-timestamps, preferring the
   newer/valid fact and saying so;
4. strict abstention — "I don't know" is a correct answer; never fabricate;
   memory ids are cited inline and validated against the retrieved
   evidence, so a fabricated citation can never escape.

Reads are sync per NS convention: the endpoint awaits the loop and returns
the synthesized answer. Each LLM call is capped by the tier's timeout
(``ASK_TIMEOUT_<LEVEL>_S``); the loop is capped by the tier's iteration
budget, so the total latency is bounded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


class AskUnavailable(RuntimeError):
    """The answering LLM was unreachable / timed out after retries."""


@dataclass(frozen=True)
class ReasoningTier:
    """One reasoning level's joint budget selection."""

    name: str
    search_limit: int      # retrieval breadth per search pass
    extra_searches: int    # LLM-directed follow-up searches (iteration cap)
    keyword_pass: bool     # grep-style exact-match pass over visible memories
    update_pass: bool      # update-language semantic pass (gated on temporal cues)
    max_answer_words: int  # output cap (instructed + clipped as a backstop)

    @property
    def llm_timeout_s(self) -> int:
        """Per-LLM-call timeout for this tier (config-driven)."""
        return int(getattr(settings, f"ask_timeout_{self.name}_s"))


REASONING_TIERS: dict[str, ReasoningTier] = {
    # minimal = single index search + direct answer, no loop.
    "minimal": ReasoningTier("minimal", search_limit=5, extra_searches=0,
                             keyword_pass=False, update_pass=False,
                             max_answer_words=80),
    "low": ReasoningTier("low", search_limit=10, extra_searches=1,
                         keyword_pass=False, update_pass=True,
                         max_answer_words=150),
    "medium": ReasoningTier("medium", search_limit=15, extra_searches=2,
                            keyword_pass=True, update_pass=True,
                            max_answer_words=250),
    # high = iterative search loop with grep-style exact-match passes.
    "high": ReasoningTier("high", search_limit=25, extra_searches=4,
                          keyword_pass=True, update_pass=True,
                          max_answer_words=400),
}

# Update-language terms appended for the forced recency pass (discipline 2).
_UPDATE_TERMS = "changed updated rescheduled moved now currently latest"

# Temporal/update cues that justify spending a full hybrid search on the
# recency pass (audit 27 #19): change verbs, "now/currently/latest/still/
# anymore/no longer/used to", and question forms like "when did …". An
# atemporal ask ("what is X's favorite color") skips the pass — appending
# update terms to an off-topic question only diluted the evidence.
_UPDATE_LANGUAGE_RE = re.compile(
    r"\b(chang(?:e[ds]?|ing)|updat(?:e[ds]?|ing)|mov(?:e[ds]?|ing)|"
    r"reschedul(?:e[ds]?|ing)|now|currently|latest|still|anymore|any more|"
    r"no longer|used to|when did)\b",
    re.IGNORECASE,
)


def _has_update_language(text: str) -> bool:
    """Whether ``text`` carries a temporal/update cue (discipline 2 gate)."""
    return bool(_UPDATE_LANGUAGE_RE.search(text or ""))

# Enumeration/counting intent (discipline 1).
_ENUMERATION_RE = re.compile(
    r"\b(how many|how much|count|number of|list (?:all|every|the)|enumerate|"
    r"all (?:the|of the)|every)\b",
    re.IGNORECASE,
)

# Small stopword set for keyword extraction — just enough to keep the
# grep pass from matching on glue words.
_STOPWORDS = frozenset(
    "the and for are was were with that this from have has had what when where "
    "which who whom how why does did doing will would could should about into "
    "than then them they their there here your you our its it's not all any "
    "list every each many much been being can may might must".split()
)

_ANSWER_ABSTAINED_NO_EVIDENCE = (
    "I don't know — no stored memories are relevant to this question."
)

# Evidence rendering caps: keep the context block bounded even at high tier.
# Fact rows keep a tight budget; passage rows (verbatim ingest chunks,
# memory_kind="passage") get the full ingest chunk size — the answer often
# sits deep inside the chunk (audit 27 #16; ingest/chunking.py
# DEFAULT_MAX_CHARS = 1500).
_EVIDENCE_CONTENT_CLIP = 500
_EVIDENCE_PASSAGE_CLIP = 1500
_EVIDENCE_MAX_ROWS = 120


def is_enumeration_question(question: str) -> bool:
    """Whether the question asks to count/enumerate (discipline 1 trigger)."""
    return bool(_ENUMERATION_RE.search(question or ""))


def extract_keywords(question: str, max_terms: int = 8) -> list[str]:
    """Naive keyword extraction for the grep-style exact pass."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{2,}", (question or "").lower())
    seen: list[str] = []
    for w in words:
        if w in _STOPWORDS or w in seen:
            continue
        seen.append(w)
        if len(seen) >= max_terms:
            break
    return seen


def _make_llm_call(tier: ReasoningTier):
    """Build the timeout-capped, lightly-retried answering-LLM callable.

    Mirrors the dreaming sweep's wrapper around the shared Gemini call path
    (``_async_call_gemini``), but raises :class:`AskUnavailable` on
    exhaustion instead of returning "" — an ask is a synchronous read and
    the caller needs an honest failure, not a silent empty answer.
    """
    from extensions.conversation_compiler.compile import _async_call_gemini

    async def call(prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(2):  # one retry — the caller is waiting
            try:
                return await asyncio.wait_for(
                    _async_call_gemini(prompt), timeout=tier.llm_timeout_s
                )
            except Exception as exc:  # timeout or transport
                last_exc = exc
                logger.warning(
                    "ask LLM call failed (attempt %d/2, tier=%s): %s",
                    attempt + 1, tier.name, exc.__class__.__name__,
                )
        raise AskUnavailable(
            f"Answering model unavailable (tier={tier.name}): {last_exc}"
        ) from last_exc

    return call


def _evidence_rows(evidence: dict, keyword_ids: list[str], enumeration: bool) -> list:
    """Select and order evidence for the prompt (audit 27 #15).

    When over the row budget, the keep-set is chosen by priority —
    keyword-pass hits first, then highest search score, then newest —
    instead of the old ascending-sort + head-truncation, which silently
    dropped the NEWEST facts (and let timestamp-less graph rows, whose
    sort key was "", eat the budget from the front).

    Survivors are then re-sorted chronologically ASCENDING for rendering
    (timestamps drive the recency/contradiction disciplines); rows with no
    timestamp sort LAST, not first. For enumeration questions the
    exact-keyword hits are listed FIRST — the discipline is
    exact-before-semantic, and list-position is how an LLM weighs evidence.
    """
    def _created(mem) -> str:
        return str(getattr(mem, "created_at", None) or "")

    rows = list(evidence.values())

    if len(rows) > _EVIDENCE_MAX_ROWS:
        kw_set = set(keyword_ids)
        # Three stable sorts, applied lowest-priority first:
        rows.sort(key=_created, reverse=True)  # newest first ("" sorts last)
        rows.sort(key=lambda m: (getattr(m, "score", None) is None,
                                 -(getattr(m, "score", None) or 0.0)))
        rows.sort(key=lambda m: m.id not in kw_set)  # keyword hits first
        rows = rows[:_EVIDENCE_MAX_ROWS]

    # Chronological ascending for the prompt; timestamp-less survivors last.
    rows.sort(key=lambda m: (0, _created(m)) if _created(m) else (1, ""))
    if enumeration and keyword_ids:
        kw = [m for m in rows if m.id in keyword_ids]
        rest = [m for m in rows if m.id not in keyword_ids]
        rows = kw + rest
    return rows


def _clip_content(content: str, budget: int) -> str:
    """Clip ``content`` to ``budget`` chars at a natural boundary.

    Prefers the last sentence boundary (". " or a newline) within budget,
    falls back to the last whitespace, and only hard-slices when the text
    is one unbroken token — the old blind ``content[:500]`` slice routinely
    cut passages mid-fact/mid-word (audit 27 #16).
    """
    if len(content) <= budget:
        return content
    window = content[:budget]
    sentence_cut = max(window.rfind(". "), window.rfind("\n"))
    if sentence_cut > 0:
        return window[: sentence_cut + 1].rstrip() + " …"
    ws_cut = window.rfind(" ")
    if ws_cut > 0:
        return window[:ws_cut].rstrip() + " …"
    return window + " …"


def _render_evidence(rows: list) -> str:
    lines = []
    for mem in rows:
        is_passage = getattr(mem, "memory_kind", None) == "passage"
        budget = _EVIDENCE_PASSAGE_CLIP if is_passage else _EVIDENCE_CONTENT_CLIP
        # Clip BEFORE flattening newlines so "\n" still marks boundaries.
        content = _clip_content((mem.memory or "").strip(), budget).replace("\n", " ")
        created = getattr(mem, "created_at", None) or "unknown time"
        category = getattr(mem, "category", None) or "uncategorized"
        lines.append(f"[{mem.id}] ({created}; {category}) {content}")
    return "\n".join(lines)


_DISCIPLINES_FULL = """Disciplines (follow strictly):
1. ENUMERATION/COUNTING: if the question asks how many / to list items, first build a
   deduplication table — group evidence rows that describe the SAME real-world item worded
   differently — then count or list the deduplicated groups. Never count raw rows.
2. RECENCY: newer memories supersede older ones. When rows describe a change ("changed",
   "rescheduled", "now", "moved to"), the newest row is the current truth.
3. CONTRADICTIONS: when two memories genuinely contradict, surface BOTH with their
   timestamps, prefer the newer/valid one, and say explicitly that you are preferring it
   because it is newer.
4. ABSTENTION: "I don't know" is a correct answer. If the evidence does not contain the
   answer, abstain — NEVER fabricate facts, dates, or memory ids.
5. CITATIONS: cite supporting memory ids inline like [<id>]. Only ids from the EVIDENCE
   list are valid citations."""

_DISCIPLINES_BRIEF = """Rules: answer ONLY from the evidence; newer memories supersede older ones; if the
evidence doesn't answer the question say you don't know (never fabricate); cite supporting
memory ids inline like [<id>]."""


def _build_prompt(
    question: str,
    evidence: dict,
    tier: ReasoningTier,
    budget: int,
    enumeration: bool,
    keyword_ids: list[str],
    keyword_scan_capped: bool = False,
) -> str:
    """Assemble the answering prompt. Verbosity scales with the tier."""
    parts = [
        "You answer a question using ONLY the user's stored memories below (the EVIDENCE list).",
        "",
        _DISCIPLINES_BRIEF if tier.name == "minimal" else _DISCIPLINES_FULL,
    ]
    if enumeration and tier.name != "minimal":
        if keyword_scan_capped:
            # Audit 27 #18: the lexical scan hit its point cap — its hits
            # are a sample, and claiming them exact/exhaustive licenses
            # confidently wrong counts.
            parts.append(
                "\nThis looks like an enumeration/counting question. A keyword scan "
                "contributed the rows listed first, but it was PARTIAL (it did not cover "
                "all stored memories) — do not treat those rows as exhaustive or complete. "
                "Build the dedup table before you count, and qualify the count if the "
                "evidence may be incomplete."
            )
        else:
            parts.append(
                "\nThis looks like an enumeration/counting question. Exact-keyword matches are "
                "listed first in the evidence. Build the dedup table before you count."
            )
    if tier.name in ("medium", "high"):
        parts.append(
            "\nThink through the evidence step by step BEFORE answering, but output ONLY "
            "the JSON object described below — no reasoning text outside it."
        )
    parts.append("\nEVIDENCE:")
    parts.append(_render_evidence(_evidence_rows(evidence, keyword_ids, enumeration)))
    parts.append(f"\nQUESTION: {question}")

    if budget > 0:
        parts.append(
            f"\nRespond with ONLY one JSON object (no prose, no code fences). Either request "
            f"ONE more search when the evidence looks incomplete — e.g. an enumeration that "
            f"may have unlisted members, or a fact that might have been updated since — with:\n"
            f'  {{"action": "search", "query": "<new search query>"}}\n'
            f"(you have {budget} search(es) left), or give the final answer with:\n"
            f'  {{"action": "answer", "answer": "<answer, at most {tier.max_answer_words} words, '
            f'memory ids cited inline like [<id>]>", "citations": ["<memory-id>", ...], '
            f'"abstained": <true when you do not know>}}'
        )
    else:
        parts.append(
            f"\nRespond with ONLY one JSON object (no prose, no code fences):\n"
            f'  {{"action": "answer", "answer": "<answer, at most {tier.max_answer_words} words, '
            f'memory ids cited inline like [<id>]>", "citations": ["<memory-id>", ...], '
            f'"abstained": <true when you do not know>}}'
        )
    return "\n".join(parts)


def _parse_llm_json(raw: str) -> dict | None:
    """Best-effort parse of the model's JSON reply (fence-tolerant)."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            out = json.loads(text[start:end + 1])
            return out if isinstance(out, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _clip_words(answer: str, max_words: int) -> str:
    """Hard output-cap backstop (the prompt already instructs the cap)."""
    words = answer.split()
    # Allow modest overshoot — clipping mid-sentence is worse than +50%.
    if len(words) <= int(max_words * 1.5):
        return answer
    return " ".join(words[: max_words]) + " …"


async def ask_memory(
    service,
    *,
    question: str,
    user_id: str,
    reasoning_level: str = "low",
    project_id: str | None = None,
    llm_call=None,
) -> dict:
    """Answer ``question`` from the caller's memories at ``reasoning_level``.

    Returns the ``AskMemoryResponse`` field set as a plain dict:
    ``{status, reasoning_level, answer, citations, abstained, searches,
    memories_considered}``. Raises :class:`AskUnavailable` when the
    answering LLM can't be reached within the tier's timeout budget, and
    ``ValueError`` on an unknown reasoning level.

    ``llm_call`` is injectable for tests; production builds the shared
    Gemini path (see ``_make_llm_call``).
    """
    tier = REASONING_TIERS.get(reasoning_level)
    if tier is None:
        raise ValueError(
            f"Invalid reasoning_level '{reasoning_level}'. "
            f"Must be one of: {list(REASONING_TIERS)}"
        )

    # Cross-source dedup (audit 27 #17): a Graphiti edge and its Qdrant twin
    # arriving in DIFFERENT passes dodge search's per-pass dedup and would
    # occupy two evidence rows. Reuse the service layer's content-match rule
    # so both dedup passes agree on what "the same fact" means.
    from memory_service import MemoryService

    searches: list[str] = []
    evidence: dict[str, object] = {}
    content_index: dict[str, str] = {}  # normalized content -> evidence id
    keyword_ids: list[str] = []

    def _merge(results) -> None:
        for r in results or []:
            if not (getattr(r, "id", None) and getattr(r, "memory", None)):
                continue
            if r.id in evidence:
                continue
            normalized = MemoryService.normalize_memory_content(r.memory)
            twin_norm = MemoryService.find_duplicate_content(
                normalized, content_index
            )
            if twin_norm is None:
                evidence[r.id] = r
                content_index[normalized] = r.id
                continue
            # Content twin: prefer the vector row (richer metadata) over
            # its graph counterpart; otherwise keep the incumbent.
            twin_id = content_index[twin_norm]
            incumbent = evidence[twin_id]
            if (
                getattr(r, "source", None) == "vector"
                and getattr(incumbent, "source", None) != "vector"
            ):
                del evidence[twin_id]
                del content_index[twin_norm]
                evidence[r.id] = r
                content_index[normalized] = r.id

    async def _semantic(query: str) -> None:
        searches.append(query)
        results = await asyncio.to_thread(
            service.search,
            query=query,
            user_id=user_id,
            project_id=project_id,
            limit=tier.search_limit,
        )
        _merge(results)

    enumeration = is_enumeration_question(question)

    # ── Discipline 1: grep-style exact pass BEFORE semantic ──
    keyword_scan_capped = False
    if tier.keyword_pass:
        terms = extract_keywords(question)
        if terms:
            searches.append("keyword: " + " ".join(terms))
            try:
                res = await asyncio.to_thread(
                    service.keyword_search,
                    user_id=user_id,
                    terms=terms,
                    project_id=project_id,
                    limit=max(20, tier.search_limit),
                )
            except Exception as e:  # non-fatal: degrade to semantic only
                logger.warning(f"ask keyword pass failed (non-critical): {e}")
                res = []
            # keyword_search returns (rows, scan_capped); tolerate a bare
            # list from older forks / test doubles.
            if isinstance(res, tuple):
                hits, keyword_scan_capped = res
            else:
                hits = res
            keyword_ids = [h.id for h in hits if getattr(h, "id", None)]
            _merge(hits)

    # ── Semantic pass (always) ──
    await _semantic(question)

    # ── Discipline 2: update-language pass, gated on temporal cues ──
    # (audit 27 #19) Runs only when the question or the first-pass evidence
    # signals the answer may have changed; otherwise the pass is skipped
    # (recorded in `skipped_passes`) and its hybrid-search cost saved.
    skipped_passes: list[str] = []
    if tier.update_pass:
        temporal = _has_update_language(question) or any(
            _has_update_language(getattr(m, "memory", None) or "")
            for m in evidence.values()
        )
        if temporal:
            await _semantic(f"{question} {_UPDATE_TERMS}")
        else:
            skipped_passes.append("update")

    # ── Strict abstention short-circuit: nothing retrieved at all ──
    if not evidence:
        return {
            "status": "ok",
            "reasoning_level": tier.name,
            "answer": _ANSWER_ABSTAINED_NO_EVIDENCE,
            "citations": [],
            "abstained": True,
            "searches": searches,
            "memories_considered": 0,
            "skipped_passes": skipped_passes,
            "_evidence_tokens": 0,
        }

    call = llm_call or _make_llm_call(tier)

    # ── Answering loop: at most extra_searches follow-ups + 1 final pass ──
    budget = tier.extra_searches
    raw = ""
    parsed: dict | None = None
    for _ in range(tier.extra_searches + 1):
        prompt = _build_prompt(question, evidence, tier, budget, enumeration,
                               keyword_ids, keyword_scan_capped)
        raw = await call(prompt)
        parsed = _parse_llm_json(raw)
        if (
            parsed is not None
            and parsed.get("action") == "search"
            and budget > 0
            and str(parsed.get("query") or "").strip()
        ):
            budget -= 1
            await _semantic(str(parsed["query"]).strip())
            continue
        break
    # The model spent its last iteration asking for another search — force
    # one final answer-only pass so the caller never gets a non-answer.
    if parsed is not None and parsed.get("action") == "search":
        prompt = _build_prompt(question, evidence, tier, 0, enumeration,
                               keyword_ids, keyword_scan_capped)
        raw = await call(prompt)
        parsed = _parse_llm_json(raw)

    if parsed is not None and "answer" in parsed:
        answer = str(parsed.get("answer") or "").strip()
        raw_citations = parsed.get("citations") or []
        citations = [str(c) for c in raw_citations if str(c) in evidence]
        abstained = bool(parsed.get("abstained")) or not answer
    else:
        # Unparseable reply: treat the raw text as the answer and recover
        # citations by scanning for evidence ids it actually mentions.
        answer = (raw or "").strip().strip("`").strip()
        citations = [mid for mid in evidence if mid in answer]
        abstained = not answer
    if not answer:
        answer = "I don't know — the evidence did not yield an answer."
        abstained = True

    # E2: measured token baseline of everything retrieved as evidence — the
    # cost a memoryless caller would have paid to read it all. Internal key
    # (leading underscore): the endpoint pops it for the savings ledger
    # before building AskMemoryResponse. Stored write-time counts mean this
    # is arithmetic, not tokenization; skipped entirely when the meter is
    # off (kill-switch = zero tokenizer calls even for legacy rows).
    from config import settings as _settings
    from savings_meter import hit_tokens

    evidence_tokens = (
        sum(hit_tokens(m) for m in evidence.values())
        if _settings.savings_meter_enabled
        else 0
    )

    return {
        "status": "ok",
        "reasoning_level": tier.name,
        "answer": _clip_words(answer, tier.max_answer_words),
        "citations": citations,
        "abstained": abstained,
        "searches": searches,
        "memories_considered": len(evidence),
        "skipped_passes": skipped_passes,
        "_evidence_tokens": evidence_tokens,
    }
