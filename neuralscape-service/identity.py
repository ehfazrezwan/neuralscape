"""Map a verified login email to a stable Neuralscape ``user_id``.

``user_id`` flows into Qdrant/Neo4j group-ids and is validated everywhere by
``schemas._ID_PATTERN`` = ``^[a-zA-Z0-9_.\\-]+$`` (max 100 chars), so it must
never contain ``@``. Two-step resolution:

1. **Override map** — ``AUTH_IDENTITY_MAP`` (email→user_id) lets known users
   keep an existing id (e.g. ``alice@example.com`` → ``alice``),
   preserving memories already stored under that id.
2. **Slug** — everyone else gets a deterministic slug of their email, so the
   same person always resolves to the same id across sessions.

The slug is lossy (``a+b@x.com`` and ``a-b@x.com`` both give ``a-b-x.com``) and
existing users' ids must never change, so ``slugify_email`` is left as is and a
collision is resolved at first login instead: when the slug is already claimed
by a different identity, the newcomer gets ``disambiguate_user_id(slug, email)``
(slug + short hash of the normalised email). See
``login_providers.provision_user_id``.
"""

from __future__ import annotations

import hashlib
import re

# Characters allowed in a user_id (mirror of schemas._ID_PATTERN's class).
_DISALLOWED = re.compile(r"[^a-zA-Z0-9_.\-]+")
_EDGE = re.compile(r"^[._\-]+|[._\-]+$")
_MAX_LEN = 100


def slugify_email(email: str) -> str:
    """Deterministically turn an email into a valid ``user_id``.

    ``alice.smith@example.com`` → ``alice.smith-example.com``.
    Any run of disallowed characters (incl. ``@``) collapses to a single ``-``;
    leading/trailing separators are trimmed; the result is capped at 100 chars.
    Falls back to a hashed id for inputs that slug to empty (e.g. all-symbol
    localparts) so we always return a non-empty, pattern-valid id.
    """
    norm = (email or "").strip().lower()
    slug = _DISALLOWED.sub("-", norm)
    slug = _EDGE.sub("", slug)
    if len(slug) > _MAX_LEN:
        slug = _EDGE.sub("", slug[:_MAX_LEN])
    if not slug:
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
        slug = f"user-{digest}"
    return slug


def derive_user_id(email: str, identity_map: dict[str, str]) -> str:
    """Resolve an email to a user_id: override map first, else a slug."""
    norm = (email or "").strip().lower()
    mapped = identity_map.get(norm)
    if mapped:
        return mapped
    return slugify_email(norm)


def disambiguate_user_id(slug: str, email: str, *, hex_len: int = 8) -> str:
    """Return ``slug`` made unique to ``email``: ``<slug>-<sha256(email)[:n]>``.

    Deterministic, pattern-valid and capped at 100 chars (the slug is trimmed
    to make room for the suffix). Only used when the plain slug is already
    claimed by someone else; never changes an existing user's id.
    """
    norm = (email or "").strip().lower()
    suffix = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:hex_len]
    head = _EDGE.sub("", slug[: _MAX_LEN - len(suffix) - 1])
    return f"{head}-{suffix}" if head else f"user-{suffix}"
