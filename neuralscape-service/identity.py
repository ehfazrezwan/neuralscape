"""Map a verified login email to a stable Neuralscape ``user_id``.

``user_id`` flows into Qdrant/Neo4j group-ids and is validated everywhere by
``schemas._ID_PATTERN`` = ``^[a-zA-Z0-9_.\\-]+$`` (max 100 chars), so it must
never contain ``@``. Two-step resolution:

1. **Override map** — ``AUTH_IDENTITY_MAP`` (email→user_id) lets known users
   keep an existing id (e.g. ``alice@example.com`` → ``alice``),
   preserving memories already stored under that id.
2. **Slug** — everyone else gets a deterministic slug of their email, so the
   same person always resolves to the same id across sessions.

The legacy slug folds every run of disallowed characters to one ``-``, so
distinct emails can share an id (``a+b@x.com`` and ``a-b@x.com``).  Under the
enforced auth profile (``NS_AUTH_PROFILE=enforced``, the default for new
deployments) a fresh identity gets :func:`collision_free_user_id` instead: the
same slug whenever the slug can be read back to exactly one email, otherwise an
id no slug can produce.  ``legacy_compat`` keeps the legacy slug byte for byte.
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


# Emails whose legacy slug reads back to exactly one address: the local part has no
# ``-`` (so the first ``-`` of the slug is the ``@``), nothing is folded, and no edge
# character is trimmed.
_SLUG_SAFE_EMAIL = re.compile(r"[a-z0-9][a-z0-9_.]*@[a-z0-9_.\-]*[a-z0-9]")
# Ids no slug can start with (a slug starts with a letter or digit): ``_x`` carries
# the address itself, hex-encoded; ``_h`` a SHA-256 digest when that would be too long.
_ENCODED_PREFIX = "_x"
_DIGEST_PREFIX = "_h"


def collision_free_ids_enabled() -> bool:
    """Whether new ids and storage namespaces use the collision-free encodings.

    On under the enforced auth profile only: ``legacy_compat`` (the default for
    upgraded deployments) keeps every id and group id it already stored.
    """
    from config import settings

    return settings.ns_auth_profile == "enforced"


def collision_free_user_id(email: str) -> str:
    """A pattern-valid ``user_id`` that differs for every distinct normalized email.

    An address whose legacy slug is unambiguous keeps that slug
    (``alice.smith@example.com`` -> ``alice.smith-example.com``).  Any other
    address gets ``_x`` + its UTF-8 hex when that fits the 100-character limit,
    else ``_h`` + its SHA-256 hex digest; a slug never starts with ``_``.
    """
    norm = (email or "").strip().lower()
    if _SLUG_SAFE_EMAIL.fullmatch(norm) and len(norm) <= _MAX_LEN:
        return norm.replace("@", "-", 1)
    encoded = norm.encode("utf-8").hex()
    if len(_ENCODED_PREFIX) + len(encoded) <= _MAX_LEN:
        return _ENCODED_PREFIX + encoded
    return _DIGEST_PREFIX + hashlib.sha256(norm.encode("utf-8")).hexdigest()


def unlinked_user_id(email: str) -> str:
    """The id a fresh, unlinked identity is offered: legacy slug or collision-free id."""
    if collision_free_ids_enabled():
        return collision_free_user_id(email)
    return slugify_email(email)


def derive_user_id(email: str, identity_map: dict[str, str]) -> str:
    """Resolve an email to a user_id: override map first, else a slug."""
    norm = (email or "").strip().lower()
    mapped = identity_map.get(norm)
    if mapped:
        return mapped
    return unlinked_user_id(norm)
