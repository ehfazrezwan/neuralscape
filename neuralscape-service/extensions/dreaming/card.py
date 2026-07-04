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
        return [l for l in card if isinstance(l, str)]
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
    distillation: only the user themself or a dictator may read it. A
    missing caller identity (local/stdio, no auth layer) is trusted like
    the rest of the local surface.
    """
    if not pool.startswith("user--"):
        return True
    if is_dictator or caller_user_id is None:
        return True
    return pool == f"user--{caller_user_id}" or pool.startswith(f"user--{caller_user_id}--")


def card_target(vault: Path, batch: PoolBatch, operator_user_id: str) -> tuple[bool, Path | None]:
    """(pool qualifies for a card, vault render path or None).

    Cards exist per **user** (private pool, no project) and per **project**
    (shared project pool). Project-scoped private pools and the global
    shared pool carry no card. A non-operator user's card is maintained in
    Redis but never rendered into the operator's vault (pool isolation).
    """
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


def _store_card(redis, pool: str, record: dict) -> None:
    try:
        redis.set(_CARD_KEY.format(pool=pool), json.dumps(record))
    except Exception:
        logger.warning("card write failed for %s (non-fatal)", pool, exc_info=True)


def _input_hash(memories: list[dict], prior_lines: list[str]) -> str:
    """Fingerprint of everything the card pass would read — unchanged
    inputs mean an unchanged card, with zero LLM spend."""
    digest = hashlib.sha256()
    for mem in sorted(memories, key=lambda m: m.get("memory_id") or ""):
        digest.update((mem.get("memory_id") or "").encode())
        digest.update(b"\x00")
        digest.update((mem.get("content") or "").strip().encode())
        digest.update(b"\x01")
    for line in prior_lines:
        digest.update(line.encode())
        digest.update(b"\x02")
    return digest.hexdigest()[:16]


# ── Vault render ────────────────────────────────────────────────────


def render_card_md(pool: str, lines: list[str], updated_at: str) -> str:
    """``Card.md`` — grammar lines verbatim (greppable, injectable as-is)."""
    return "\n".join([
        "---",
        "title: Card",
        f"pool: {pool}",
        f"updated: {updated_at}",
        f"lines: {len(lines)}",
        "---",
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
) -> dict:
    """Maintain one pool's identity card from the staged batch + prior card.

    Returns ``{"status": "updated"|"stable"|"unchanged"|"skipped", ...}``:
    *unchanged* = inputs identical, LLM skipped; *stable* = LLM ran but
    reproduced the prior card (updated_at preserved); *updated* = the
    card actually changed.
    """
    qualifies, file_path = card_target(vault, batch, operator_user_id)
    if not qualifies:
        return {"status": "skipped", "reason": "pool carries no card"}

    live = [
        m for m in batch.memories
        if not m.get("dream_tombstoned") and (m.get("content") or "").strip()
    ]
    prior = load_card(redis, batch.pool) or {}
    prior_lines = sanitize_card_lines(prior.get("lines"))
    if not live and not prior_lines:
        return {"status": "skipped", "reason": "no memories and no prior card"}

    fingerprint = _input_hash(live, prior_lines)
    if prior_lines and prior.get("input_hash") == fingerprint:
        # Nothing the pass reads has changed → the card cannot change.
        if not dry_run and file_path is not None and not file_path.exists():
            _write_card_file(file_path, batch.pool, prior_lines, prior.get("updated_at") or "")
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


def _write_card_file(path: Path, pool: str, lines: list[str], updated_at: str) -> None:
    from extensions.conversation_compiler.obsidian_writer import _atomic_write

    try:
        _atomic_write(path, render_card_md(pool, lines, updated_at))
    except Exception:
        logger.warning("card render failed for %s (non-fatal)", path, exc_info=True)
