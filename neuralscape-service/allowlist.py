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


def normalize_email(email: str | None) -> str:
    """Lowercase + strip. Returns '' for falsy input."""
    return (email or "").strip().lower()


def email_domain(email: str) -> str:
    """The domain part of a normalized email, or '' if malformed.

    Requires exactly one ``@`` with a non-empty local-part AND domain, and no
    whitespace or control characters anywhere, so ``@example.com``, ``a@`` and
    ``x@evil.com@example.com`` are all rejected — important because this feeds
    the allowlist's domain gate. Kept in step with the Supabase hook SQL
    (``supabase/allowlist_hook.sql``), which also rejects more than one ``@``.
    """
    norm = normalize_email(email)
    if any(ch.isspace() or not ch.isprintable() for ch in norm):
        return ""
    if norm.count("@") != 1:
        return ""
    local, _, domain = norm.partition("@")
    if not local or not domain:
        return ""
    return domain


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
    norm = normalize_email(email)
    # Require a well-formed address: non-empty local-part AND domain. This
    # rejects malformed inputs like "@example.com" that would otherwise slip
    # through the domain gate (email_domain returns '' for those).
    domain = email_domain(norm)
    if not domain:
        return False
    if not allowed_domains and not email_allowlist:
        # Fail closed: an unconfigured allowlist must not admit the world.
        return False
    if norm in email_allowlist:
        return True
    return domain in allowed_domains
