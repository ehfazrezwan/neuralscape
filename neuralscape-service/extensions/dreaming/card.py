"""Identity card — the pinned, grammar-constrained grounding artifact (B4).

One card per user pool and per project pool: max :data:`CARD_MAX_LINES`
lines, every line matching ``^(IDENTITY|ATTRIBUTE|RELATIONSHIP|INSTRUCTION): .+$``
— the grammar is enforced in code (non-conforming LLM lines are dropped,
overlong cards truncated), never trusted to the prompt.

Maintained by the dreaming sweep: a small LLM pass per qualifying pool
updates the card from the staged batch plus the prior card. The pass is
**additive/corrective, not regenerative** — the prompt instructs the
model to keep stable lines verbatim, and two code-level stability locks
back it up: an input fingerprint skips the LLM entirely when neither the
staged memories nor the prior card changed, and an LLM output identical
to the prior card keeps the prior ``updated_at`` (no churn).

Storage (three surfaces, ZERO of them searchable memories):

1. pinned Redis artifact — ``dreaming:card:{pool}`` (no TTL);
2. vault render — ``Me/Card.md`` for the operator's user pool,
   ``Projects/<pid>/Card.md`` for project pools (other users' cards stay
   Redis-only: pool isolation, spec §4.2);
3. read surfaces — the ``get_card`` MCP tool and
   ``GET /v1/extensions/dreaming/card?pool=`` for session-start injection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .consolidate import PoolBatch
from .prompts import parse_json_object

logger = logging.getLogger(__name__)

CARD_MAX_LINES = 40
CARD_LINE_RE = re.compile(r"^(IDENTITY|ATTRIBUTE|RELATIONSHIP|INSTRUCTION): .+$")

_CARD_KEY = "dreaming:card:{pool}"


CARD_PROMPT = """\
You maintain the pinned identity card of one memory pool — the compact
grounding artifact an agent injects at session start. Update the PRIOR
CARD using the MEMORIES below. Emit STRICT JSON only — no prose, no
markdown fences.

Grammar — every line MUST match exactly one of these shapes:
  IDENTITY: <who this user/project fundamentally is>
  ATTRIBUTE: <a stable trait, preference, or fact>
  RELATIONSHIP: <a person / team / system relationship>
  INSTRUCTION: <a standing instruction agents must follow>

Rules:
- ADDITIVE AND CORRECTIVE, never regenerative: reproduce prior lines
  VERBATIM unless a memory contradicts one (rewrite that line) or makes
  it obsolete (drop it). Do not rephrase lines that are still true.
- Add new lines only for card-worthy material: stable, identity-level
  facts — never episodic details, one-off events, or transient state.
- At most {max_lines} lines; fewer is better. One fact per line, third
  person, no markdown, no numbering, no lines outside the grammar.

PRIOR CARD:
{prior_card}

MEMORIES (id | category | content):
{memories_block}

Output schema:
{{"card": ["IDENTITY: ...", "ATTRIBUTE: ...", "INSTRUCTION: ..."]}}
"""


# ── Grammar enforcement (in code, not in faith) ─────────────────────


def sanitize_card_lines(lines) -> list[str]:
    """Drop non-conforming lines, dedupe, truncate to the 40-line ceiling."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines or []:
        if not isinstance(raw, str):
            continue
        line = raw.strip().lstrip("-*• ").strip()
        if not CARD_LINE_RE.match(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= CARD_MAX_LINES:
            break
    return out


def parse_card_response(raw: str) -> list[str]:
    """Card lines from the LLM response: JSON ``{"card": [...]}`` first,
    bare lines as the fallback. Grammar filtering happens in sanitize."""
    obj = parse_json_object(raw)
    card = obj.get("card")
    if isinstance(card, list):
        return [line for line in card if isinstance(line, str)]
    return (raw or "").splitlines()


# ── Pool targeting ──────────────────────────────────────────────────


def resolve_card_pool(
    *, user_id: str | None = None, project_id: str | None = None, pool: str | None = None
) -> str | None:
    """Pool key for a card read: explicit pool > project pool > user pool."""
    if pool:
        return pool
    if project_id:
        return f"shared--project--{project_id}"
    if user_id:
        return f"user--{user_id}"
    return None


def card_read_allowed(pool: str, caller_user_id: str | None, *, is_dictator: bool = False) -> bool:
    """May this caller read this pool's card?

    Project cards are team artifacts — any authenticated caller reads
    them. A ``user--<uid>`` card is that user's private identity
    distillation: only the user themself or a dictator may read it.

    ``caller_user_id`` must be the caller's *effective* identity resolved
    by the surface (verified token identity when present, else the
    claimed/request user_id, else the configured default — the same
    precedence every other read path uses). ``None`` denies: this guard
    never treats a missing identity as trust.
    """
    if not pool.startswith("user--"):
        return True
    if is_dictator:
        return True
    if not caller_user_id:
        return False
    return pool == f"user--{caller_user_id}" or pool.startswith(f"user--{caller_user_id}--")


def card_target(vault: Path, batch: PoolBatch, operator_user_id: str) -> tuple[bool, Path | None]:
    """(pool qualifies for a card, vault render path or None).

    Cards exist per **user** (private pool, no project) and per **project**
    (shared project pool). Project-scoped private pools and the global
    shared pool carry no card. A non-operator user's card is maintained in
    Redis but never rendered into the operator's vault (pool isolation).

    WT6: reference workspaces (workspace ≠ None and ≠ "memory") are excluded
    — the card structurally cannot see reference content, preventing the
    leak that triggered this partition (trading books poisoning "you are a trader").
    """
    # Gate reference workspaces out of card eligibility entirely (WT6).
    if batch.workspace and batch.workspace != "memory":
        return False, None

    if batch.visibility == "shared" and batch.project_id:
        from .librarian import pool_dir

        target = pool_dir(vault, batch, operator_user_id)
        return True, (target / "Card.md" if target is not None else None)
    if batch.visibility == "private" and not batch.project_id:
        if batch.owner_user_id == operator_user_id:
            return True, vault / "Me" / "Card.md"
        return True, None
    return False, None


# ── Redis persistence (pinned artifact — never a memory) ────────────


def load_card(redis, pool: str) -> dict | None:
    """The pinned card record ``{lines, updated_at, input_hash}`` or None."""
    try:
        raw = redis.get(_CARD_KEY.format(pool=pool))
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.warning("card read failed for %s (non-fatal)", pool, exc_info=True)
        return None


def build_card_view(pool: str, caller_user_id: str | None, *, is_dictator: bool, redis) -> dict:
    """The shared read contract for both surfaces (REST route + MCP tool).

    Centralizes the authorization gate and the response shape so the two
    surfaces cannot silently diverge. ``status`` is one of ``forbidden``
    (private card, wrong caller), ``not_found`` (no card yet), or ``ok``
    (with ``pool`` / ``lines`` / ``card`` / ``updated_at``).
    """
    if not card_read_allowed(pool, caller_user_id, is_dictator=is_dictator):
        return {"status": "forbidden"}
    data = load_card(redis, pool)
    lines = (data or {}).get("lines") or []
    if not lines:
        return {"status": "not_found"}
    return {
        "status": "ok",
        "pool": pool,
        "lines": lines,
        "card": "\n".join(lines),
        "updated_at": (data or {}).get("updated_at"),
    }


def _store_card(redis, pool: str, record: dict) -> None:
    try:
        redis.set(_CARD_KEY.format(pool=pool), json.dumps(record))
    except Exception:
        logger.warning("card write failed for %s (non-fatal)", pool, exc_info=True)


def _input_hash(memories: list[dict], prior_lines: list[str]) -> str:
    """Fingerprint of everything the card pass would read — unchanged
    inputs mean an unchanged card, with zero LLM spend.

    Covers every per-memory field the prompt renders (id, category,
    created_at, content — see ``render_memories_block``) plus the prior
    card, so no prompt-visible change can be skipped as "unchanged".
    """
    digest = hashlib.sha256()
    for mem in sorted(memories, key=lambda m: m.get("memory_id") or ""):
        for key in ("memory_id", "category", "created_at", "content"):
            digest.update(str(mem.get(key) or "").strip().encode())
            digest.update(b"\x00")
        digest.update(b"\x01")
    for line in prior_lines:
        digest.update(line.encode())
        digest.update(b"\x02")
    return digest.hexdigest()[:16]


# ── Vault render ────────────────────────────────────────────────────


def render_card_md(pool: str, lines: list[str], updated_at: str) -> str:
    """``Card.md`` — grammar lines verbatim (greppable, injectable as-is)."""
    from okf import translate as okf_translate

    return "\n".join([
        okf_translate.concept_frontmatter(
            page_kind="card",
            title="Card",
            description="Pinned identity card — grammar-constrained grounding lines.",
            timestamp=updated_at,
            extensions={"pool": pool, "updated": updated_at, "lines": len(lines)},
        ),
        "",
        "# Card",
        "",
        *lines,
        "",
        "---",
        "Maintained by the dreaming sweep — edits here are overwritten.",
        "",
    ])


# ── The card pass ───────────────────────────────────────────────────


async def update_card(
    batch: PoolBatch,
    llm_call,
    *,
    redis,
    vault: Path,
    operator_user_id: str,
    dry_run: bool,
    render_files: bool = True,
) -> dict:
    """Maintain one pool's identity card from the staged batch + prior card.

    Returns ``{"status": "updated"|"stable"|"unchanged"|"skipped", ...}``:
    *unchanged* = inputs identical, LLM skipped; *stable* = LLM ran but
    reproduced the prior card (updated_at preserved); *updated* = the
    card actually changed.

    ``render_files=False`` keeps the card Redis-only (the sweep passes
    ``vault_pages_enabled`` here — an operator who disabled vault output
    must not find Card.md files appearing under the vault path anyway).
    """
    qualifies, file_path = card_target(vault, batch, operator_user_id)
    if not qualifies:
        return {"status": "skipped", "reason": "pool carries no card"}
    if not render_files:
        file_path = None

    live = [
        m for m in batch.memories
        if not m.get("dream_tombstoned") and (m.get("content") or "").strip()
    ]
    # WT6: belt-and-braces — even in the memory pool, order personal_fact/
    # preference/interaction first so imported reference content (if somehow
    # present) can't dominate the 40-line card. Card-eligible pools shouldn't
    # have reference workspace rows (card_target gates them), but this ordering
    # ensures domain_knowledge imports can't swamp a user's actual preferences.
    _personal_categories = {"personal_fact", "preference", "interaction"}
    live.sort(
        key=lambda m: (
            0 if m.get("category") in _personal_categories else 1,
            m.get("created_at") or "",
        )
    )
    prior = load_card(redis, batch.pool) or {}
    prior_lines = sanitize_card_lines(prior.get("lines"))
    if not live and not prior_lines:
        return {"status": "skipped", "reason": "no memories and no prior card"}

    fingerprint = _input_hash(live, prior_lines)
    if prior_lines and prior.get("input_hash") == fingerprint:
        # Nothing the pass reads has changed → the card cannot change.
        # Still re-assert the vault render (missing or hand-edited files
        # converge back to the pinned artifact — Redis is authoritative).
        if not dry_run and file_path is not None:
            _write_card_file(
                file_path, batch.pool, prior_lines,
                prior.get("updated_at") or "",
                only_if_stale=True,
            )
        return {"status": "unchanged", "lines": len(prior_lines)}

    from .prompts import render_memories_block

    raw = await llm_call(CARD_PROMPT.format(
        max_lines=CARD_MAX_LINES,
        prior_card="\n".join(prior_lines) or "(no card yet)",
        memories_block=render_memories_block(live, include_strength=False),
    ))
    lines = sanitize_card_lines(parse_card_response(raw))
    if not lines:
        if not prior_lines:
            return {"status": "skipped", "reason": "no card-worthy lines"}
        lines = prior_lines  # LLM failure/garbage → the prior card stands

    changed = lines != prior_lines
    updated_at = (
        datetime.now(timezone.utc).isoformat()
        if changed or not prior.get("updated_at")
        else prior["updated_at"]
    )
    if not dry_run:
        _store_card(redis, batch.pool, {
            "lines": lines,
            "updated_at": updated_at,
            # Hash against the lines being stored (they are next sweep's
            # prior) so an identical staged batch skips the LLM next time.
            "input_hash": _input_hash(live, lines),
        })
        if file_path is not None:
            _write_card_file(file_path, batch.pool, lines, updated_at)
    return {"status": "updated" if changed else "stable", "lines": len(lines)}


def _write_card_file(
    path: Path, pool: str, lines: list[str], updated_at: str, *, only_if_stale: bool = False
) -> None:
    """Render Card.md. With ``only_if_stale``, write only when the file is
    missing or drifted from the expected render (byte-idempotent no-op
    otherwise — the steady state touches nothing)."""
    from extensions.conversation_compiler.obsidian_writer import _atomic_write

    try:
        rendered = render_card_md(pool, lines, updated_at)
        if only_if_stale and path.exists():
            try:
                if path.read_text(encoding="utf-8") == rendered:
                    return
            except Exception:
                pass  # unreadable → rewrite
        _atomic_write(path, rendered)
    except Exception:
        logger.warning("card render failed for %s (non-fatal)", path, exc_info=True)
