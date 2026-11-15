# Authentication & Login Providers

How Neuralscape authenticates humans and machines, and how to switch the
human-login mechanism per deployment with a single env var.

## Fail-closed by default

The service **refuses to start** unless one of these is set:

| Setting | Use |
|---|---|
| `NEURALSCAPE_USER_TOKEN_SECRET` | Per-user HMAC tokens (recommended; also enables the OAuth connector with `NEURALSCAPE_PUBLIC_URL`) |
| `NEURALSCAPE_API_KEY` | Legacy single shared key (deprecated, still honoured; acts as one bound identity, see [Legacy shared key](#legacy-shared-key-one-identity)) |
| `NEURALSCAPE_AUTH_DISABLED=true` | No authentication at all. Trusted local machine only |

With neither secret set and no opt-out, startup fails with
`Authentication is not configured. Set one of: ...`, and `BearerAuthMiddleware`
answers every non-public request with 503 instead of passing it through. The
opt-out logs one `AUTHENTICATION IS DISABLED` WARNING at startup and is ignored
whenever either secret is set. Public paths are `/health`, `/health/live`,
`/.well-known/*` and `/oauth/*`.

The MCP **stdio** transport (`python mcp_server.py`, launched as a local
subprocess by Claude Code) has no network listener and no bearer layer; it is
reachable only by the process that spawned it, so this rule does not apply to
it. The ARQ workers likewise open no inbound port.

## The model: two independent layers

1. **The MCP OAuth layer** — Neuralscape is its own OAuth 2.1 Authorization
   Server (`oauth.py`): Dynamic Client Registration, `/oauth/authorize`,
   PKCE-bound auth codes, `/oauth/token`, refresh, and the `.well-known/oauth-*`
   discovery metadata. This is what Claude Cowork / claude.ai talk to (the
   connector UI is OAuth-only). **It does not change** when you switch login
   providers.

   Refresh tokens are single-use and bound to the client that got them. Each
   refresh returns a new token in the same family. If an already-used refresh
   token is presented again, the whole family is revoked and the user reconnects.
   There is no grace window, so a client that retries a refresh after a lost
   response also has its session revoked (OAuth 2.0 Security BCP reuse
   detection). The single-use and revocation markers live in Redis; if Redis is
   down, `/oauth/token` answers 503 `temporarily_unavailable` and issues nothing.

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

## Legacy shared key: one identity

`NEURALSCAPE_API_KEY` is **deprecated** (no removal date is set). Every response
to a caller using it carries an `X-Neuralscape-Deprecation` header. Prefer
per-user tokens.

The shared key acts as **one user**, not as anyone the caller names:

| Setting | Meaning |
|---|---|
| `NEURALSCAPE_API_KEY_USER_ID` | The user the key acts as. Empty means `DEFAULT_USER_ID`. |
| `NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID` | `true` restores the old behaviour (key may act as any body/query `user_id`). Default `false`. A WARNING is logged at startup when on. |

With the default, the middleware binds the key to that user exactly like a
per-user token: a request whose body or query `user_id` names anyone else gets
**HTTP 400**; a request with no `user_id` acts as the bound user; MCP ignores
`arguments["user_id"]`; task status is readable only for tasks that user
enqueued. The key holder is also an admin when `DICTATOR_USER_IDS` is empty and
the bound user is the default user.

> **Upgrade note.** Before upgrading, operators who use the shared key on
> behalf of a specific user (for example a plugin whose configured user id is
> `alice`) must set `NEURALSCAPE_API_KEY_USER_ID=alice`, otherwise the key is
> bound to `DEFAULT_USER_ID` and calls naming `alice` fail with 400. If one key
> serves several users, set `NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID=true`
> until each user has their own token (`scripts/issue_user_token.py`).

## Token revocation

A per-user token is valid until it expires (forever, for one issued with
`--no-expiry`). Revoking works on the token's SHA-256 digest; the token itself
is never stored or printed. Two lists are enforced together (their union):

1. **Live, no restart: the Redis set `ns:auth:revoked_tokens`.**

   ```bash
   printf '%s' "$LEAKED_TOKEN" | python scripts/revoke_token.py --redis      # revoke
   printf '%s' "$LEAKED_TOKEN" | python scripts/revoke_token.py --unrevoke   # undo
   python scripts/revoke_token.py --digest <sha256-hex> --unrevoke          # undo without the token
   python scripts/revoke_token.py --list                                    # show revoked digests
   ```

   The script uses `REDIS_URL`. Every API process re-reads the set at most every
   `REVOCATION_REFRESH_INTERVAL_SECONDS` (default 5; 0 = on every request), so a
   revocation takes effect on all instances within that many seconds, and a
   request costs no Redis round trip while the cached snapshot is fresh.
2. **Static fallback: `NEURALSCAPE_REVOKED_TOKENS=<digest>[,<digest>...]`**
   (digest printed by `python scripts/revoke_token.py`; needs a restart). It is
   always honoured, whatever state Redis is in.

A revoked token is rejected with 401 on REST and MCP, and cannot be pasted into
the OAuth consent screen. If the env list cannot be parsed, per-user tokens are
rejected (fail closed).

**When Redis is unreachable** (exact semantics):

- The env list is still enforced.
- After at least one successful read, the in-process snapshot keeps being
  enforced: everything revoked up to the outage stays revoked. A revocation
  made during the outage is not seen until Redis is back (at most one interval
  after it returns).
- At cold start with Redis down there is no snapshot yet: only the env list is
  enforced, and tokens are NOT refused on account of the outage (fail open for
  the Redis list only, so a Redis restart does not lock every user out). This
  is logged as a warning (`token revocation: Redis unreachable ... env list
  only`). Operators who need a hard guarantee put the digest in the env list.
- After a failed read the next attempt waits one interval, so an outage adds one
  short connect timeout (`REVOCATION_REDIS_TIMEOUT_SECONDS`, default 1) per
  interval, not per request. The warning is logged once per outage window.

Staleness bound: a revocation is enforced at most
`REVOCATION_REFRESH_INTERVAL_SECONDS` after it is written (plus the request
that triggers the refresh), per API process.

**OAuth sessions.** Authorization codes, access tokens and refresh tokens carry a
`src` claim: the digest of the credential the human logged in with (the pasted
token) or, for Google/Supabase logins, of the identity (`identity_fingerprint`,
printed by `revoke_token.py --identity <provider>:<user_id>`). `src` is carried
across refresh rotation. Revoking that digest (live or env list) ends every
session minted from it: access tokens are 401, refresh and code grants are
`invalid_grant`. Revoking an OAuth access or refresh token directly works by its
own digest; revoking a refresh token also revokes its family (the same
`oauth:revoked_family:<fam>` key #495 uses on reuse), so its successors die too.
Sessions minted before this change carry no `src` and are covered only by their
own digests. Unrevoking a refresh token does not resurrect its family.

Limits, plainly:

- It covers the exact token string only (a fresh token for the same user is a
  different token), plus every OAuth session derived from it (below).
- Rotate `NEURALSCAPE_USER_TOKEN_SECRET` to invalidate every token.
- Existing non-expiring tokens keep verifying unless revoked or the secret rotates.

`scripts/issue_user_token.py --days N` requires `N >= 1`; a token with no
expiry needs the explicit `--no-expiry` flag.

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
| `POST /oauth/authorize` | Token-paste submit (when `AUTH_PROVIDER=token`, or any provider with `AUTH_ALLOW_TOKEN_PASTE=true`) |
| `POST /oauth/link` | First-login link decision (continue-as-new / claim existing) |
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

- **Refuses to start with `Authentication is not configured`** — set `NEURALSCAPE_USER_TOKEN_SECRET` (or the legacy `NEURALSCAPE_API_KEY`), or `NEURALSCAPE_AUTH_DISABLED=true` on a trusted local machine. See [Fail-closed by default](#fail-closed-by-default).
- **Every request returns 503 `Authentication is not configured`** — the app is being served without its startup validation and no secret is set; same fix as above.
- **Boots with a config error naming `GOOGLE_OAUTH_*` / `SUPABASE_*` / `AUTH_ALLOWED_DOMAINS`** — that provider's required env vars are missing; the validator lists every gap.
- **"This account is not authorized"** — verified email isn't in the allowlist (env for `google`; the `signup_email_domains`/`addresses` tables for `supabase`).
- **Google `redirect_uri_mismatch`** — the URI in Google Cloud must be exactly `https://<host>/oauth/google/callback`.
- **Supabase login spins on the finish page** — the redirect isn't allow-listed in the Supabase dashboard, or the start/finish pages aren't same-origin.
- **A user "lost" their memories after switching** — they resolved to a slug; add `email:old_user_id` to `AUTH_IDENTITY_MAP`.
- **Cowork can't connect at all** — that's the MCP OAuth layer, not the provider: check `NEURALSCAPE_PUBLIC_URL` + `NEURALSCAPE_USER_TOKEN_SECRET` are set and `/.well-known/oauth-protected-resource` returns 200. See [COWORK.md](COWORK.md).
