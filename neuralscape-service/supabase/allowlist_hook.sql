-- Neuralscape — Supabase "Before User Created" allowlist hook
-- =============================================================
-- Used when AUTH_PROVIDER=supabase. This is the canonical gate that decides
-- WHO may sign up (and therefore who Neuralscape will mint a user_id for):
-- the Neuralscape service trusts that any Supabase session it receives belongs
-- to an allowed account because this hook blocked everyone else at signup.
--
-- Policy: DEFAULT-DENY. A signup is allowed only when the verified email's
-- domain is in `signup_email_domains` OR the exact address is in
-- `signup_email_addresses`. Everything else is rejected with HTTP 403. This
-- mirrors the service's fail-closed env allowlist (allowlist.py).
--
-- Apply: run this in the Supabase SQL editor (or `supabase db push`), then
-- register the function as the Before-User-Created hook (see bottom of file).
-- Payload shape ref: event->'user'->>'email'  (Supabase docs, 2026).

-- ── allowlist tables ────────────────────────────────────────────────────

create table if not exists public.signup_email_domains (
  domain     text primary key,
  note       text,
  created_at timestamptz not null default now()
);

create table if not exists public.signup_email_addresses (
  email      text primary key,
  note       text,
  created_at timestamptz not null default now()
);

comment on table public.signup_email_domains is
  'Allowlisted email domains for Supabase signup (AUTH_PROVIDER=supabase). Lowercase, no leading @.';
comment on table public.signup_email_addresses is
  'Allowlisted individual email addresses for Supabase signup. Lowercase.';

-- ── lock the tables down ────────────────────────────────────────────────
-- Supabase exposes the public schema over PostgREST to the `anon` and
-- `authenticated` roles, so without this anyone holding the public anon key
-- could read the allowlist or INSERT their own address into it. RLS on with no
-- policy for those roles = deny; the explicit revoke removes the table
-- privileges Supabase grants to them by default. The hook itself is SECURITY
-- DEFINER (runs as the table owner, which bypasses RLS); the policy below also
-- lets supabase_auth_admin read the tables directly. Idempotent.

alter table public.signup_email_domains   enable row level security;
alter table public.signup_email_addresses enable row level security;

revoke all on public.signup_email_domains   from anon, authenticated, public;
revoke all on public.signup_email_addresses from anon, authenticated, public;

drop policy if exists signup_email_domains_auth_admin_read on public.signup_email_domains;
create policy signup_email_domains_auth_admin_read on public.signup_email_domains
  for select to supabase_auth_admin using (true);

drop policy if exists signup_email_addresses_auth_admin_read on public.signup_email_addresses;
create policy signup_email_addresses_auth_admin_read on public.signup_email_addresses
  for select to supabase_auth_admin using (true);

-- ── seed (REPLACE with your org's domain) ───────────────────────────────
-- Auto-allow your whole Workspace domain.
insert into public.signup_email_domains (domain, note)
values ('example.com', 'Replace with your Google Workspace domain')
on conflict (domain) do nothing;

-- One-off guests go here, e.g.:
-- insert into public.signup_email_addresses (email, note)
-- values ('contractor@gmail.com', 'Q3 contractor') on conflict (email) do nothing;

-- ── the hook function ───────────────────────────────────────────────────

-- NOTE: locals are v_-prefixed; a bare `email`/`domain` is ambiguous with the
-- column names inside the EXISTS subqueries and makes every call raise.
create or replace function public.hook_restrict_signup(event jsonb)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_email  text;
  v_domain text;
begin
  v_email := lower(event->'user'->>'email');

  -- Require a well-formed address: a non-empty local-part AND domain. The
  -- `@` must not be the first char (empty local-part like "@example.com") nor
  -- absent, and the domain part must be non-empty. (OAuth signups arrive with
  -- an email already verified by the upstream IdP, e.g. Google; the service
  -- additionally checks `email_verified` on the session JWT.)
  if v_email is null or v_email = ''
     or length(v_email) - length(replace(v_email, '@', '')) <> 1   -- exactly one '@'
     or position('@' in v_email) <= 1
     or split_part(v_email, '@', 2) = ''
  then
    return pg_catalog.jsonb_build_object(
      'error', pg_catalog.jsonb_build_object(
        'message', 'A valid email address is required to sign in.',
        'http_code', 403
      )
    );
  end if;

  v_domain := split_part(v_email, '@', 2);

  -- Exact-address allowlist wins, then domain allowlist.
  if exists (select 1 from public.signup_email_addresses a where a.email = v_email)
     or exists (select 1 from public.signup_email_domains d where d.domain = v_domain)
  then
    return '{}'::jsonb;  -- allow
  end if;

  return pg_catalog.jsonb_build_object(
    'error', pg_catalog.jsonb_build_object(
      'message', 'This account is not authorized to access Neuralscape.',
      'http_code', 403
    )
  );
end;
$$;

-- ── grants (auth admin runs the hook; nobody else may call it) ──────────

grant execute on function public.hook_restrict_signup to supabase_auth_admin;
revoke execute on function public.hook_restrict_signup from authenticated, anon, public;

-- The hook reads the allowlist tables as supabase_auth_admin; grant it access.
grant usage on schema public to supabase_auth_admin;
grant select on public.signup_email_domains   to supabase_auth_admin;
grant select on public.signup_email_addresses  to supabase_auth_admin;

-- ── enable the hook ─────────────────────────────────────────────────────
-- The Postgres hook is registered in the Dashboard (there is no public SQL
-- API for it as of 2026):
--   Authentication → Hooks → "Before User Created" → Postgres →
--   select  public.hook_restrict_signup  → Enable.
-- (Self-hosted: set
--   GOTRUE_HOOK_BEFORE_USER_CREATED_ENABLED=true
--   GOTRUE_HOOK_BEFORE_USER_CREATED_URI=pg-functions://postgres/public/hook_restrict_signup
--  in the auth service env.)
