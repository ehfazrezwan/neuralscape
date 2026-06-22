# Authentication & Login Providers

How Neuralscape authenticates humans and machines, and how to switch the
human-login mechanism per deployment with a single env var.

## The model: two independent layers

1. **The MCP OAuth layer** — Neuralscape is its own OAuth 2.1 Authorization
   Server (`oauth.py`): Dynamic Client Registration, `/oauth/authorize`,
   PKCE-bound auth codes, `/oauth/token`, refresh, and the `.well-known/oauth-*`
   discovery metadata. This is what Claude Cowork / claude.ai talk to (the
   connector UI is OAuth-only). **It does not change** when you switch login
   providers.

2. **The human-login step** — the one screen inside `/oauth/authorize` where a
   person proves who they are. This is the *only* thing `AUTH_PROVIDER` swaps.

A login provider's whole job is to turn an inbound login into a **verified,
allowlisted `user_id`** (or a rejection). Everything downstream — codes,
access/refresh tokens, `BearerAuthMiddleware`, `user_id` scoping — is identical
for every provider.

> **Machines are unaffected.** Pre-issued admin HMAC tokens
> (`scripts/issue_user_token.py`) remain valid `Authorization: Bearer`
> credentials no matter what `AUTH_PROVIDER` is. CLI, CI, and e2e keep working.

## Choosing a provider

```
AUTH_PROVIDER = token | google | supabase
```

| Value | Login UX | Allowlist source | Use it for |
|-------|----------|------------------|------------|
| `token` (default) | Paste an admin-issued HMAC token once | n/a (admin issues tokens) | local/dev, single-team, automation-only |
| `google` | "Sign in with Google" (OIDC) | **env** (`AUTH_ALLOWED_DOMAINS` / `AUTH_EMAIL_ALLOWLIST`) | public installs without a DB |
| `supabase` | "Sign in with Google" via Supabase | **Supabase** Before-User-Created hook (DB table) + optional env | managed, dashboard-editable allowlist |

A non-`token` provider requires the OAuth AS to be on:
`NEURALSCAPE_PUBLIC_URL` **and** `NEURALSCAPE_USER_TOKEN_SECRET` must be set.
Startup validation fails loudly if a provider's required vars are missing.

### Both options on one screen

When `google` or `supabase` is active, the consent page shows **both** a "Sign in
with Google" button **and** the admin-token paste box — so tokens issued by
`issue_user_token.py` remain a valid login (service accounts, CI, users without a
Google identity), and admin tokens also still work directly as
`Authorization: Bearer` credentials without visiting the page at all. Set
`AUTH_ALLOW_TOKEN_PASTE=false` for a provider-only screen (the paste box is
hidden and `POST /oauth/authorize` rejects token submits). It's always shown for
`AUTH_PROVIDER=token`.

## Identity: email → `user_id`

`user_id` flows into Qdrant/Neo4j group-ids and is validated by
`schemas._ID_PATTERN` = `^[a-zA-Z0-9_.\-]+$` (max 100), so it can never contain
`@`. Resolution (`identity.py`):

1. **Override map** — `AUTH_IDENTITY_MAP="email:user_id,..."` pins a known user
   to an existing id, **preserving memories already stored under that id**.
2. **Slug** — everyone else gets a deterministic slug:
   `alice.smith@example.com` → `alice.smith-example.com`.

> **Migration:** if memories are already keyed by short ids (e.g. `alice`),
> map those people in `AUTH_IDENTITY_MAP` before flipping a provider on, or
> they'll resolve to fresh slugs and appear to "lose" their history.

## Allowlist semantics (`allowlist.py`)

A login is accepted only when the email is **verified** AND
(`domain ∈ AUTH_ALLOWED_DOMAINS` OR `email ∈ AUTH_EMAIL_ALLOWLIST`).
Both lists are comma-separated and case-insensitive; a leading `@` on a domain
is tolerated. **An empty allowlist denies everyone (fail-closed).**

- `google`: this env allowlist is the sole gate (must be configured).
- `supabase`: the Supabase hook is the canonical gate; the env allowlist is
  applied as an *extra* gate only when configured.

---

## Setup: `AUTH_PROVIDER=token` (default)

Nothing new. Issue a token and hand it to the user (they paste it once on the
consent screen, or use it directly as a Bearer token):

```bash
docker compose exec neuralscape python scripts/issue_user_token.py --user alice --days 30
```

## Setup: `AUTH_PROVIDER=google`

1. **Google Cloud Console → APIs & Services → Credentials.**
   - Configure the OAuth **consent screen** as **External**, then **Publish**.
     Scopes are `openid email profile` (non-sensitive) → **no verification
     review required**.
   - Create an **OAuth client ID → Web application**.
   - Authorized redirect URI: `https://<your-host>/oauth/google/callback`
   - Copy the client ID + secret.
2. **Env:**
   ```env
   AUTH_PROVIDER=google
   NEURALSCAPE_PUBLIC_URL=https://<your-host>
   NEURALSCAPE_USER_TOKEN_SECRET=<already set>
   GOOGLE_OAUTH_CLIENT_ID=...
   GOOGLE_OAUTH_CLIENT_SECRET=...
   AUTH_ALLOWED_DOMAINS=example.com
   # AUTH_EMAIL_ALLOWLIST=guest@gmail.com
   # AUTH_IDENTITY_MAP=alice@example.com:alice
   ```
3. Restart. Connecting in Cowork now shows Google sign-in; only allowlisted
   verified emails get through.

## Setup: `AUTH_PROVIDER=supabase`

1. **Supabase → Authentication → Providers → Google:** enable it (Supabase
   manages its own Google OAuth client). Add
   `https://<your-host>/oauth/supabase/callback` to the project's redirect
   allow-list (and the Site URL / redirect settings).
2. **Allowlist hook:** run [`neuralscape-service/supabase/allowlist_hook.sql`](neuralscape-service/supabase/allowlist_hook.sql)
   in the SQL editor. It creates `signup_email_domains` / `signup_email_addresses`,
   seeds a placeholder `example.com` (replace it), and defines
   `public.hook_restrict_signup` (default-deny). Then **Authentication → Hooks → Before User Created →
   Postgres → select `hook_restrict_signup` → Enable.**
   - Add/remove access later with plain SQL — no redeploy:
     ```sql
     insert into public.signup_email_domains (domain, note) values ('partner.com','Partner org');
     insert into public.signup_email_addresses (email, note) values ('guest@gmail.com','One-off');
     ```
3. **Env:**
   ```env
   AUTH_PROVIDER=supabase
   NEURALSCAPE_PUBLIC_URL=https://<your-host>
   NEURALSCAPE_USER_TOKEN_SECRET=<already set>
   SUPABASE_URL=https://<project>.supabase.co
   SUPABASE_ANON_KEY=<anon/publishable key>
   # SUPABASE_JWT_SECRET=<legacy HS256 secret>   # omit → verify via project JWKS
   # AUTH_ALLOWED_DOMAINS=...   # optional extra gate on top of the hook
   # AUTH_IDENTITY_MAP=alice@example.com:alice
   ```
   Leave `SUPABASE_JWT_SECRET` empty to verify session JWTs against the
   project's asymmetric JWKS (recommended; supports key rotation). Set it only
   if your project still uses the legacy HS256 shared secret.
4. Restart.

### Supabase flow internals (for debugging)

`GET /oauth/authorize` renders a page that calls supabase-js
`signInWithOAuth({provider:'google', redirectTo: …/oauth/supabase/callback})`.
Google → Supabase → `GET /oauth/supabase/callback` (browser): supabase-js
exchanges the PKCE `code` for a session (verifier in localStorage, same-origin),
then POSTs the session JWT to `POST /oauth/supabase/callback`, which verifies it
and continues the MCP flow. The start and finish pages **must be same-origin**
(both under `NEURALSCAPE_PUBLIC_URL`) for the PKCE exchange to work.

---

## Endpoint reference

| Path | Purpose |
|------|---------|
| `GET /.well-known/oauth-authorization-server` | RFC 8414 AS metadata |
| `GET /.well-known/oauth-protected-resource[/mcp]` | RFC 9728 PRM |
| `POST /oauth/register` | Dynamic Client Registration |
| `GET /oauth/authorize` | Consent step (delegates to the active provider) |
| `POST /oauth/authorize` | Token-paste submit (only when `AUTH_PROVIDER=token`) |
| `GET /oauth/google/callback` | Google OIDC redirect target |
| `GET /oauth/supabase/callback` | Supabase browser finisher (renders) |
| `POST /oauth/supabase/callback` | Supabase JWT verify → continue |
| `POST /oauth/token` | Code/refresh → access token |

## Testing

```bash
cd neuralscape-service
uv run pytest tests/test_auth_providers.py tests/test_oauth.py tests/test_auth.py -v
```

Unit tests cover the allowlist, identity derivation, login-state signing, and
both providers (Supabase via real HS256 JWTs, Google via a stubbed exchange).
The Supabase **browser** PKCE round-trip can only be smoke-tested against a
real deployment.

## Troubleshooting

- **Boots with a config error naming `GOOGLE_OAUTH_*` / `SUPABASE_*` / `AUTH_ALLOWED_DOMAINS`** — that provider's required env vars are missing; the validator lists every gap.
- **"This account is not authorized"** — verified email isn't in the allowlist (env for `google`; the `signup_email_domains`/`addresses` tables for `supabase`).
- **Google `redirect_uri_mismatch`** — the URI in Google Cloud must be exactly `https://<host>/oauth/google/callback`.
- **Supabase login spins on the finish page** — the redirect isn't allow-listed in the Supabase dashboard, or the start/finish pages aren't same-origin.
- **A user "lost" their memories after switching** — they resolved to a slug; add `email:old_user_id` to `AUTH_IDENTITY_MAP`.
- **Cowork can't connect at all** — that's the MCP OAuth layer, not the provider: check `NEURALSCAPE_PUBLIC_URL` + `NEURALSCAPE_USER_TOKEN_SECRET` are set and `/.well-known/oauth-protected-resource` returns 200. See [COWORK.md](COWORK.md).
