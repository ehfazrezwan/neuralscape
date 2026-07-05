"""E4 — custom extraction instructions (per-user and per-project).

The lightweight 80% of knowledge adapters: an operator supplies plain-text
guidance ("always tag decisions with the ADR number when present", "extract
customer names verbatim") that is appended to the extraction prompt as a
clearly-delimited OPERATOR GUIDANCE addendum. It steers *what* gets
extracted and *how facts are phrased*; it can NEVER change the output
contract (see ``prompts.append_operator_guidance`` — the addendum carries
an explicit prompt-injection guard, and the fence-tolerant parser is the
code-level backstop).

Two scopes, composed when both exist (project first, then user):

- **user** — self-set via ``PUT /v1/settings/extraction-instructions``
  (no ``project_id``); applies to every extraction for that user.
- **project** — dictator-only (mirrors the standards write gate); applies
  to every extraction targeting that ``project_id``, whoever triggers it.

Instructions apply to conversation extraction (``extract_and_store``, the
conversation-compiler flush) AND ingest distillation
(``extract_facts_only``), and *compose* with knowledge adapters: the
adapter owns the base prompt, the operator addendum rides after it.

Token budget (``EXTRACTION_INSTRUCTIONS_MAX_TOKENS``, default 2,000) is
enforced at save time — a stored instruction is always within budget.

Storage: small JSON records in Redis (no TTL — settings, not sessions):

- ``ns:extraction-instructions:user:{user_id}``
- ``ns:extraction-instructions:project:{project_id}``
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

from index_format import estimate_tokens

logger = logging.getLogger(__name__)

USER_KEY = "ns:extraction-instructions:user:{user_id}"
PROJECT_KEY = "ns:extraction-instructions:project:{project_id}"

_redis = None
_redis_lock = threading.Lock()


class _LocalJsonStore:
    """redis-shaped (get/set/delete) whole-value store over one JSON file.

    Solo engine (unit 5): extraction instructions are two tiny JSON records
    keyed by the same strings as the Redis keys — a local file is a direct
    port. Writes are atomic (tmp + replace) under a process-wide lock; solo
    is a single daemon, so no cross-process coordination is needed.
    """

    def __init__(self, path):
        from pathlib import Path

        self._path = Path(path).expanduser()
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            return json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}
        except Exception:  # noqa: BLE001 — a corrupt file reads as empty
            logger.warning("extraction-settings file unreadable; treating as empty")
            return {}

    def _write(self, data: dict) -> None:
        import os

        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        os.replace(tmp, self._path)

    def get(self, key: str):
        with self._lock:
            return self._load().get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            data = self._load()
            data[key] = value
            self._write(data)

    def delete(self, key: str) -> None:
        with self._lock:
            data = self._load()
            if key in data:
                del data[key]
                self._write(data)


_local_store: _LocalJsonStore | None = None


def _get_redis():
    """The settings store: Redis in team mode, a local JSON file in solo.

    Kept under the historical name because the three public functions take
    an injectable ``redis=`` handle whose only contract is get/set/delete.
    """
    global _redis, _local_store
    from config import settings

    if settings.ns_mode == "solo":
        if _local_store is None:
            with _redis_lock:
                if _local_store is None:
                    _local_store = _LocalJsonStore(settings.extraction_settings_path)
        return _local_store
    if _redis is None:
        with _redis_lock:
            if _redis is None:
                import redis as redis_lib

                _redis = redis_lib.Redis.from_url(
                    settings.redis_url, socket_timeout=3, socket_connect_timeout=3
                )
    return _redis


def _key(*, user_id: str | None = None, project_id: str | None = None) -> str:
    if project_id:
        return PROJECT_KEY.format(project_id=project_id)
    if user_id:
        return USER_KEY.format(user_id=user_id)
    raise ValueError("user_id or project_id required")


def instruction_tokens(text: str | None) -> int:
    """Token cost of an instruction block — real tiktoken count when the
    encoder loads, len/4 heuristic otherwise (same counter the summarizer
    and assembler use, so the save-time budget matches serve-time math)."""
    if not text:
        return 0
    from savings_meter import _get_encoder

    enc = _get_encoder()
    if enc is None:
        return estimate_tokens(text)
    return len(enc.encode(text))


def validate_instructions(text: str) -> tuple[int, str | None]:
    """(token count, error message or None). Budget-enforced at save time."""
    from config import settings

    tokens = instruction_tokens(text)
    cap = int(settings.extraction_instructions_max_tokens)
    if tokens > cap:
        return tokens, (
            f"extraction_instructions is {tokens} tokens; the budget is "
            f"{cap} tokens (EXTRACTION_INSTRUCTIONS_MAX_TOKENS)."
        )
    return tokens, None


def set_instructions(
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    instructions: str,
    updated_by: str,
    redis=None,
) -> dict:
    """Store (or clear, when empty/whitespace) one scope's instructions.

    The caller has already authorized the write (project scope is
    dictator-only — enforced at the endpoint, mirroring
    ``_authorize_standard_write``). Raises ``ValueError`` over budget.
    """
    r = redis if redis is not None else _get_redis()
    key = _key(user_id=user_id, project_id=project_id)
    text = (instructions or "").strip()
    if not text:
        r.delete(key)
        return {"instructions": None, "tokens": 0, "updated_at": None, "updated_by": None}
    tokens, error = validate_instructions(text)
    if error:
        raise ValueError(error)
    record = {
        "instructions": text,
        "tokens": tokens,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": updated_by,
    }
    r.set(key, json.dumps(record, ensure_ascii=False))
    return record


def get_instructions(
    *, user_id: str | None = None, project_id: str | None = None, redis=None
) -> dict | None:
    """One scope's stored record, or None. Never raises on a down Redis."""
    try:
        r = redis if redis is not None else _get_redis()
        raw = r.get(_key(user_id=user_id, project_id=project_id))
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) and data.get("instructions") else None
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 — settings reads must degrade
        logger.warning("extraction-instructions read failed (non-fatal)", exc_info=True)
        return None


def resolve_instructions(
    user_id: str | None, project_id: str | None = None, redis=None
) -> str | None:
    """The composed operator guidance for one extraction: project-wide
    guidance first (it binds everyone working the project), then the
    user's own. None when neither is set or the feature is disabled.
    Best-effort — a down Redis yields None, never a failed extraction."""
    from config import settings

    if not settings.extraction_instructions_enabled:
        return None
    if not user_id and not project_id:
        return None
    parts: list[str] = []
    try:
        r = redis if redis is not None else _get_redis()
        if project_id:
            rec = get_instructions(project_id=project_id, redis=r)
            if rec:
                parts.append(f"[project guidance]\n{rec['instructions']}")
        if user_id:
            rec = get_instructions(user_id=user_id, redis=r)
            if rec:
                parts.append(f"[user guidance]\n{rec['instructions']}")
    except Exception:  # noqa: BLE001
        logger.warning("extraction-instructions resolve failed (non-fatal)", exc_info=True)
        return None
    return "\n\n".join(parts) or None


def _reset_for_tests() -> None:
    global _redis
    _redis = None
