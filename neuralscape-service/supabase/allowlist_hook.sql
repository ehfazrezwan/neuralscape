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

-- ── seed (REPLACE with your org's domain) ───────────────────────────────
-- Auto-allow your whole Workspace domain.
insert into public.signup_email_domains (domain, note)
values ('example.com', 'Replace with your Google Workspace domain')
on conflict (domain) do nothing;

-- One-off guests go here, e.g.:
-- insert into public.signup_email_addresses (email, note)
-- values ('contractor@gmail.com', 'Q3 contractor') on conflict (email) do nothing;

-- ── the hook function ───────────────────────────────────────────────────

create or replace function public.hook_restrict_signup(event jsonb)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  email  text;
  domain text;
begin
  email := lower(event->'user'->>'email');

  -- No email on the account → reject (we scope memory by email-derived id).
  if email is null or email = '' or position('@' in email) = 0 then
    return jsonb_build_object(
      'error', jsonb_build_object(
        'message', 'An email address is required to sign in.',
        'http_code', 403
      )
    );
  end if;

  domain := split_part(email, '@', 2);

  -- Exact-address allowlist wins, then domain allowlist.
  if exists (select 1 from public.signup_email_addresses a where a.email = email)
     or exists (select 1 from public.signup_email_domains d where d.domain = domain)
  then
    return '{}'::jsonb;  -- allow
  end if;

  return jsonb_build_object(
    'error', jsonb_build_object(
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
