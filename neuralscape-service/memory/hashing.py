"""Content hashing, timestamp parsing, and small pure helpers.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import hashlib

from datetime import datetime, timezone
from config import settings

def _parse_expires_at(value) -> datetime | None:
    """Parse an `expires_at` payload value to an aware UTC datetime.

    Accepts ISO-8601 strings (with or without a trailing `Z`), `datetime`
    instances (naive treated as UTC), or anything else returns None. Used by
    the expiry cron — comparing raw strings is unsafe across mixed offsets
    (`Z` vs `+00:00` vs `-05:00` won't sort lexicographically).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # datetime.fromisoformat doesn't accept the literal 'Z' suffix until
    # Python 3.11 fully; normalize defensively.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def content_hash(content: str) -> str:
    """Canonical content hash for write-path dedup (audit 27 #6).

    The service-layer writers of ``payload["hash"]`` — store_raw, the
    conversation batch path, checkpoints, and the dreaming rewrite (which
    re-embeds new text in place) — all route through this helper; new
    writers should too. The exact-dedup cron groups rows by this value and
    hard-deletes "duplicates", so a stale or divergent hash silently
    corrupts dedup in both directions.
    """
    return hashlib.md5(content.encode()).hexdigest()


def _infer_project_id(content: str) -> str | None:
    """Try to infer a project_id from memory content by matching known project slugs.

    Slugs are deployment-specific and come from KNOWN_PROJECT_SLUGS.
    """
    content_lower = content.lower()
    for slug in settings.known_projects:
        if slug in content_lower:
            return slug
    return None


def _created_at_key(value) -> datetime:
    """Chronological sort key for a created_at payload value.

    Parses ISO-8601 (any offset — mem0-written rows may carry non-UTC
    offsets, where lexicographic string comparison missorts). Unparseable /
    missing values sort first.
    """
    if value:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)
