"""Email allowlist for federated login.

A login is accepted when the email is *verified* AND either:
  * its domain is in ``AUTH_ALLOWED_DOMAINS`` (auto-whitelist a Workspace),
    or
  * the exact address is in ``AUTH_EMAIL_ALLOWLIST`` (one-off guests).

Used directly by the ``google`` login provider. For the ``supabase`` provider
the canonical gate is Supabase's Before-User-Created hook (a DB table), but the
same check is applied as optional defense-in-depth when the env allowlist is
also configured.

Pure functions, no web-framework deps — trivially unit-testable.
"""

from __future__ import annotations

import re


def normalize_email(email: str | None) -> str:
    """Lowercase + strip. Returns '' for falsy input."""
    return (email or "").strip().lower()


# Characters that may never appear anywhere in an address: ASCII/C1 controls,
# space, NBSP, soft hyphen, Unicode separators (U+2028/2029, U+3000, ...) and
# invisible format characters (zero-width, bidi marks, BOM). The SAME class is
# used by ``supabase/allowlist_hook.sql`` (which cannot hold NUL); keep in step.
_FORBIDDEN = re.compile(
    "[\u0000-\u0020\u007f-\u00a0\u00ad\u180e\u1680\u2000-\u200f"
    "\u2028-\u202f\u205f-\u2064\u2066-\u206f\u3000\ufeff]"
)


def email_domain(email: str | None) -> str:
    """The (lowercased) domain of an address, or '' if malformed.

    The RAW value is validated (no stripping): exactly one ``@``, a non-empty
    local-part AND domain, and no whitespace/control/invisible characters
    anywhere. So ``@example.com``, ``a@``, ``x@evil.com@example.com`` and
    ``x@example.com`` followed by a newline are all rejected. This feeds the
    allowlist's domain gate and matches the Supabase hook SQL
    (``supabase/allowlist_hook.sql``).
    """
    raw = email or ""
    if _FORBIDDEN.search(raw) or raw.count("@") != 1:
        return ""
    local, _, domain = raw.partition("@")
    if not local or not domain:
        return ""
    return domain.lower()


def is_email_allowed(
    email: str | None,
    *,
    email_verified: bool,
    allowed_domains: set[str],
    email_allowlist: set[str],
) -> bool:
    """True iff the (verified) email passes the domain OR exact-address gate.

    An unverified email is *never* allowed — an attacker who controls an IdP
    account with an unverified address must not be able to claim a whitelisted
    domain. When neither ``allowed_domains`` nor ``email_allowlist`` is
    configured the gate is closed (deny-all) rather than open: an empty
    allowlist means "nobody", never "everybody".
    """
    if not email_verified:
        return False
    # Validate the raw value: padding/embedded whitespace is denied, not trimmed.
    domain = email_domain(email)
    if not domain:
        return False
    norm = normalize_email(email)
    if not allowed_domains and not email_allowlist:
        # Fail closed: an unconfigured allowlist must not admit the world.
        return False
    if norm in email_allowlist:
        return True
    return domain in allowed_domains
