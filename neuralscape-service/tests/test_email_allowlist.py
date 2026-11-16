"""Single-'@' email allowlist rule.

The cases mirror ``integration_tests/test_supabase_allowlist_hook_pg.py`` (the
Supabase hook SQL); keep the two tables in step.
"""

import pytest

from allowlist import email_domain, is_email_allowed

DOMAINS = {"corp.example"}
ADDRESSES = {"guest@elsewhere.example"}


@pytest.mark.parametrize(
    ("email", "allowed"),
    [
        # allowed
        ("x@corp.example", True),  # listed domain
        ("X@CORP.EXAMPLE", True),  # uppercase domain, case-folded
        ("anyone@corp.example", True),
        ("Guest@Elsewhere.example", True),  # listed address, case-insensitive
        ("  x@corp.example  ", True),  # surrounding whitespace stripped
        # denied: not listed
        ("stranger@not-listed.example", False),
        ("other@elsewhere.example", False),  # the address is listed, not its domain
        ("x@sub.corp.example", False),  # exact domain match only
        # denied: malformed
        ("", False),
        (None, False),
        ("@example.com", False),  # empty local part
        ("nodomain@", False),  # empty domain
        ("@corp.example", False),
        # denied: more than one '@'
        ("x@corp.example@evil.example", False),
        ("x@evil.example@corp.example", False),  # the original bypass
        ("a@@corp.example", False),
        ("@@", False),
        # denied: whitespace / control characters inside
        ("x @corp.example", False),
        ("x@corp .example", False),
        ("x\n@corp.example", False),
        ("x@corp.example\x00", False),
    ],
)
def test_single_at_gate(email, allowed):
    assert (
        is_email_allowed(email, email_verified=True, allowed_domains=DOMAINS, email_allowlist=ADDRESSES)
        is allowed
    )


def test_listed_multi_at_address_is_still_denied():
    # SQL: a listed 'a@b@corp.example' is denied too.
    assert not is_email_allowed(
        "a@b@corp.example",
        email_verified=True,
        allowed_domains=set(),
        email_allowlist={"a@b@corp.example"},
    )


@pytest.mark.parametrize("email", ["x@evil.com@example.com", "a@@b.com", "x\t@b.com", "x@b.com\x00"])
def test_email_domain_rejects(email):
    assert email_domain(email) == ""


def test_email_domain_ok():
    assert email_domain("A@B.com") == "b.com"
