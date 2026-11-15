"""The Supabase signup allowlist hook, applied to a real PostgreSQL (sec-07).

``supabase/allowlist_hook.sql`` is the gate that decides who may sign up when
``AUTH_PROVIDER=supabase``. This applies the unmodified script to a scratch
database on the gate's PostgreSQL and checks, as the roles Supabase uses:

* ``anon`` and ``authenticated`` (what PostgREST connects as) can neither read
  nor write the two allowlist tables;
* the hook default-denies an unlisted email, malformed addresses, allows a
  listed address and a listed domain, and runs as ``supabase_auth_admin``;
* the hook is not callable by ``anon``/``authenticated``;
* re-applying the script is harmless (idempotent) and keeps all of the above.

Not covered (no Supabase stack on the gate): GoTrue actually invoking the hook
on a signup, and PostgREST's own role switching (``SET ROLE`` stands in for it).
Needs a PostgreSQL role allowed to CREATE DATABASE and CREATE ROLE (the
disposable gate instance is superuser).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

SQL_FILE = Path(__file__).resolve().parents[1] / "supabase" / "allowlist_hook.sql"
ROLES = ("anon", "authenticated", "supabase_auth_admin")
TABLES = ("signup_email_domains", "signup_email_addresses")


@pytest.fixture()
def hook_db(pg_dsn, unique_ns):
    """A scratch database with the stand-in Supabase roles and the script applied."""

    name = "nsit_hook_" + uuid.uuid4().hex[:10]
    admin = psycopg.connect(pg_dsn, autocommit=True)
    try:
        for role in ROLES:  # cluster-wide and shared: create only when missing
            exists = admin.execute("select 1 from pg_roles where rolname = %s", (role,)).fetchone()
            if not exists:
                admin.execute(sql.SQL("create role {} nologin").format(sql.Identifier(role)))
        admin.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
    except Exception:
        admin.close()
        raise

    def drop():
        try:
            admin.execute(sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(name)))
        finally:
            admin.close()

    unique_ns.on_cleanup(drop)

    dsn = make_conninfo(pg_dsn, dbname=name)
    conn = psycopg.connect(dsn, autocommit=True)
    unique_ns.on_cleanup(conn.close)
    # Supabase's default privileges give anon/authenticated everything on new
    # public tables; reproduce that so the REVOKE in the script is what is tested.
    conn.execute("alter default privileges in schema public grant all on tables to anon, authenticated")
    conn.execute("grant usage on schema public to anon, authenticated")
    conn.execute(SQL_FILE.read_text(encoding="utf-8"))
    return conn


def _as(conn, role, query):
    """Run ``query`` as ``role`` (like PostgREST does), always resetting the role."""

    conn.execute(sql.SQL("set role {}").format(sql.Identifier(role)))
    try:
        cur = conn.execute(query)
        return cur.fetchall() if cur.description else None
    finally:
        conn.execute("reset role")


def _hook(conn, email, role="supabase_auth_admin"):
    event = json.dumps({"user": {"email": email}})
    conn.execute(sql.SQL("set role {}").format(sql.Identifier(role)))
    try:
        return conn.execute("select public.hook_restrict_signup(%s::jsonb)", (event,)).fetchone()[0]
    finally:
        conn.execute("reset role")


def _denied(result):
    return isinstance(result, dict) and result.get("error", {}).get("http_code") == 403


@pytest.mark.parametrize("role", ["anon", "authenticated"])
@pytest.mark.parametrize("table", TABLES)
def test_public_roles_cannot_touch_the_allowlist_tables(hook_db, role, table):
    col = "domain" if table == "signup_email_domains" else "email"
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _as(hook_db, role, f"select * from public.{table}")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _as(hook_db, role, f"insert into public.{table} ({col}) values ('attacker.example')")


def test_rls_is_enabled_on_both_tables(hook_db):
    rows = hook_db.execute(
        "select relname, relrowsecurity from pg_class where relname = any(%s)", (list(TABLES),)
    ).fetchall()
    assert sorted(rows) == [(t, True) for t in sorted(TABLES)]


def test_rls_alone_denies_even_if_privileges_were_granted_back(hook_db):
    """Defence in depth: with the table privilege restored, RLS (no policy) still hides every row."""

    hook_db.execute("grant select, insert on public.signup_email_domains to anon")
    assert _as(hook_db, "anon", "select * from public.signup_email_domains") == []
    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # new row violates row-level security
        _as(hook_db, "anon", "insert into public.signup_email_domains (domain) values ('attacker.example')")


def test_hook_default_denies_unlisted_and_malformed_emails(hook_db):
    assert _denied(_hook(hook_db, "stranger@not-listed.example"))
    assert _denied(_hook(hook_db, ""))
    assert _denied(_hook(hook_db, "@example.com"))
    assert _denied(_hook(hook_db, "nodomain@"))
    assert _denied(_hook(hook_db, None))


def test_hook_denies_addresses_with_more_than_one_at_sign(hook_db):
    """``x@corp.example@evil.example`` must not be judged by its middle part."""

    hook_db.execute("insert into public.signup_email_domains (domain) values ('corp.example')")
    hook_db.execute("insert into public.signup_email_addresses (email) values ('a@b@corp.example')")
    for email in ("x@corp.example@evil.example", "x@evil.example@corp.example", "a@b@corp.example", "@@", "a@@corp.example"):
        assert _denied(_hook(hook_db, email)), email
    assert _hook(hook_db, "x@corp.example") == {}


def test_hook_function_pins_an_empty_search_path(hook_db):
    config = hook_db.execute(
        "select proconfig from pg_proc where proname = 'hook_restrict_signup'"
    ).fetchone()[0]
    assert config == ["search_path=\"\""]


def test_hook_allows_a_listed_address_and_a_listed_domain(hook_db):
    hook_db.execute("insert into public.signup_email_addresses (email) values ('guest@elsewhere.example')")
    hook_db.execute("insert into public.signup_email_domains (domain) values ('corp.example')")

    assert _hook(hook_db, "Guest@Elsewhere.example") == {}  # case-insensitive
    assert _hook(hook_db, "anyone@corp.example") == {}
    assert _denied(_hook(hook_db, "other@elsewhere.example"))  # the address is listed, not its domain
    assert _denied(_hook(hook_db, "x@sub.corp.example"))  # exact domain match only


def test_hook_reads_the_tables_when_run_by_the_auth_admin_role(hook_db):
    """The hook must keep working for supabase_auth_admin after the lockdown."""

    hook_db.execute("insert into public.signup_email_domains (domain) values ('seeded.example')")
    assert _hook(hook_db, "me@seeded.example", role="supabase_auth_admin") == {}
    # and the role can read the rows directly through the explicit policy
    rows = _as(hook_db, "supabase_auth_admin", "select domain from public.signup_email_domains")
    assert ("seeded.example",) in rows


@pytest.mark.parametrize("role", ["anon", "authenticated"])
def test_public_roles_cannot_call_the_hook(hook_db, role):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _hook(hook_db, "me@example.com", role=role)


def test_reapplying_the_script_is_idempotent_and_keeps_the_lockdown(hook_db):
    hook_db.execute(SQL_FILE.read_text(encoding="utf-8"))
    hook_db.execute(SQL_FILE.read_text(encoding="utf-8"))
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _as(hook_db, "anon", "select * from public.signup_email_addresses")
    assert _hook(hook_db, "ok@example.com") == {}  # the seeded example.com domain survives
    assert _denied(_hook(hook_db, "stranger@not-listed.example"))
