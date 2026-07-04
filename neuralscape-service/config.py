import os
from pathlib import Path
from urllib.parse import urlparse

from arq.connections import RedisSettings
from pydantic import field_validator
from pydantic_settings import BaseSettings

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


class Settings(BaseSettings):
    # Gemini (direct Google AI Studio)
    google_api_key: str = ""
    gemini_llm_model: str = "gemini-3.1-flash-lite"
    gemini_llm_fallback_model: str = "gemini-2.5-flash"
    gemini_embedder_model: str = "gemini-embedding-001"

    # ── LLM gateway (OpenAI-compatible) ───────────────────────────────
    # When enabled, the LLM + embedder (and the graphiti reranker) route
    # through an OpenAI-compatible gateway (e.g. an internal Vertex-via-ADC
    # gateway) instead of Google AI Studio — fixing AI Studio's 503 throttling.
    # The same base_url + key serve all three; model tags differ (the embedder
    # needs the provider-prefixed `google-vertex/...` tag to hit Vertex rather
    # than AI Studio). graphiti's OpenAI clients read the base_url from the
    # OPENAI_BASE_URL env var (the mem0 adapter doesn't thread base_url
    # through), which get_mem0_config sets when this flag is on.
    llm_gateway_enabled: bool = False
    llm_gateway_base_url: str = ""  # e.g. https://llm-gateway.example.com (/v1 appended if absent)
    llm_gateway_api_key: str = ""
    # The gateway only fronts Google/Anthropic models on Vertex (no OpenAI
    # models), so every model tag carries the `google-vertex/` provider prefix —
    # the bare tag routes elsewhere and doesn't enforce the strict json_schema
    # graphiti's entity extraction depends on.
    llm_gateway_llm_model: str = "google-vertex/gemini-3.1-flash-lite"
    llm_gateway_llm_fallback_model: str = "google-vertex/gemini-2.5-flash"
    llm_gateway_embedder_model: str = "google-vertex/gemini-embedding-001"
    # Model for graphiti entity/edge extraction. A stronger model does NOT help
    # the residual structured-output flakiness — that's the gateway's incomplete
    # OpenAI strict-json_schema support for Gemini, not model quality (2.5-flash
    # was slower and no better than flash-lite). Kept configurable for when the
    # gateway adds strict structured output.
    llm_gateway_graphiti_model: str = "google-vertex/gemini-3.1-flash-lite"
    # Whether graphiti (graph extraction) ALSO routes through the gateway. Default
    # False → graphiti stays on AI Studio for clean/complete extraction (native
    # Gemini handles structured output; the gateway's strict json_schema support
    # for Gemini is incomplete → flaky/partial graphs). Set True once the gateway
    # supports strict structured output. Independent of llm_gateway_enabled, which
    # routes the vector path (and is what stabilizes the API event loop).
    llm_gateway_graphiti_enabled: bool = False

    # Neo4j
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "memory"

    # Graphiti options
    store_raw_episode_content: bool = True
    update_communities: bool = False

    # Qdrant vector store
    qdrant_url: str | None = None  # e.g. "http://localhost:6333" — if set, uses Qdrant server mode
    qdrant_on_disk: bool = True
    qdrant_path: str = "~/.neuralscape/qdrant"  # only used when qdrant_url is not set
    qdrant_collection: str = "neuralscape_memories"

    # Redis / ARQ
    redis_url: str = "redis://localhost:6379"
    arq_queue_name: str = "neuralscape:queue"
    # Slow Graphiti graph writes run on their OWN queue so they can't starve
    # fast vector writes / reads. This is a DEDICATED queue by default, consumed
    # by `arq worker.GraphWorkerSettings` — that worker MUST be running (see the
    # deploy note / docker-compose neuralscape-graph-worker) or graph-enrichment
    # jobs queue unconsumed. To collapse back onto the single main worker, set
    # GRAPH_QUEUE_NAME=neuralscape:queue and register process_graph_enrichment
    # on WorkerSettings.
    graph_queue_name: str = "neuralscape:graph"
    # Bulk document/file ingestion runs on its OWN queue so a folder/zip ingest
    # (chunking + Docling parse + LLM fact extraction) can't starve latency-
    # sensitive vector writes/reads on the main queue. Consumed by
    # `arq worker.IngestWorkerSettings` — that worker MUST be running (see the
    # docker-compose neuralscape-ingest-worker) or ingest/connector-sync jobs
    # queue unconsumed. To collapse back onto the main worker, set
    # INGEST_QUEUE_NAME=neuralscape:queue and register the ingest tasks on
    # WorkerSettings.
    ingest_queue_name: str = "neuralscape:ingest"
    arq_max_retries: int = 3
    arq_job_timeout: int = 300  # 5 min max per task

    # LLM retry (exponential backoff for transient 503/429 errors)
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0  # seconds
    llm_retry_max_delay: float = 30.0  # seconds

    # Dedup cron
    dedup_similarity_threshold: float = 0.95
    dedup_batch_size: int = 100
    dedup_cron_hours: set = {0, 6, 12, 18}

    # ── Ask / reasoning tiers (roadmap C3) ────────────────────────────
    # Per-tier cap on a single answering-LLM call (seconds). The total ask
    # budget is bounded by this times the tier's iteration cap (see
    # ask.REASONING_TIERS) — higher tiers may loop through follow-up
    # searches, so they get a roomier per-call cap too.
    ask_timeout_minimal_s: int = 20
    ask_timeout_low_s: int = 40
    ask_timeout_medium_s: int = 75
    ask_timeout_high_s: int = 120

    # ── Queue visibility (roadmap C4) ─────────────────────────────────
    # Window (seconds) of recently-enqueued tasks aggregated by
    # GET /v1/queue/status. Matches ARQ's default keep_result TTL (3600s):
    # older results have expired out of Redis anyway and would only ever
    # report as "expired".
    queue_status_window_s: int = 3600
    # When set, workers POST a small {"event": "queue.empty", ...} JSON to
    # this URL whenever a finished job leaves its queue empty — so
    # ingest-then-query flows can stop polling per task. Empty = off.
    # Only absolute http(s) URLs are contacted; redirects are never
    # followed; delivery is a fire-and-forget 5s-capped daemon thread
    # (see webhooks.py).
    webhook_queue_empty_url: str = ""

    # ── Observability: SSE live stream (roadmap E1) ───────────────────
    # When enabled, memory events (memory_stored, dream actions applied,
    # insights stored, checkpoint batches) are mirrored onto Redis pub/sub
    # channels and served to authenticated callers at GET /v1/stream.
    # Publishing is fire-and-forget: a down Redis never breaks a write.
    event_stream_enabled: bool = True

    # ── Observability: token-economics telemetry (roadmap E2) ─────────
    # The honest meter. When enabled: token_estimate upgrades to REAL
    # tiktoken counts at write time, every recall op records a measured
    # {baseline, served, overhead, net} savings event to an append-only
    # per-user Redis stream, and index_only/timeline responses carry a
    # compact savings line. `net` is SIGNED and may go negative — the
    # meter never overclaims. Kill-switch: when False there are ZERO
    # tokenizer calls anywhere on the hot path (write-time stamping falls
    # back to the cheap len/4 heuristic) and no ledger writes.
    savings_meter_enabled: bool = True
    # tiktoken encoding used for all real token counts.
    savings_tokenizer: str = "o200k_base"
    # Approximate (~) cap on each per-user ledger stream (XADD maxlen).
    savings_ledger_maxlen: int = 100_000
    # Heuristic multiplier for the CLEARLY-LABELED-ESTIMATED
    # `rederivation_savings_estimate` field: tokens an agent would burn
    # re-deriving/re-discovering a fact from sources ≈ multiplier × the
    # fact's stored token count. Never blended into the measured headline.
    savings_rederivation_multiplier: float = 10.0

    # ── Session summarizer slots + context assembler (roadmap E3) ─────
    # Per-session rolling summaries, maintained on the FAST worker queue as
    # conversation writes cross message-count thresholds. Two slots per
    # session — `short` (refreshed every ~short_every messages, capped at
    # short_max_tokens) and `long` (every ~long_every, long_max_tokens) —
    # each REPLACED on refresh via recursive compression (new summary =
    # prior summary + messages since). Slots + the raw message buffer live
    # in Redis only (TTL'd), never as searchable memory rows — see
    # session_summarizer.py for the rationale.
    session_summary_enabled: bool = True
    session_summary_short_every: int = 20
    session_summary_long_every: int = 60
    session_summary_short_max_tokens: int = 1000
    session_summary_long_max_tokens: int = 4000
    # Rolling per-session message buffer: keep at most this many recent
    # messages (LTRIM), and expire all per-session keys after this many days.
    session_buffer_max_messages: int = 400
    session_ttl_days: int = 30

    # ── Data-layer connectors ─────────────────────────────────────────
    # When enabled, the service hosts connectors (Notion/Drive/MCP/REST),
    # stores their credentials encrypted in the vault, and runs a periodic
    # sync. `vault_key` is a Fernet key (generate with
    # `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
    connectors_enabled: bool = False
    vault_key: str = ""  # NEURALSCAPE_VAULT_KEY — required when connectors_enabled
    connector_sync_cron_hours: int = 6  # interval (hours) the sync cron fires

    # ── Document/file ingestion ───────────────────────────────────────
    # Rich formats (PDF, MS Office docx/xlsx/pptx, HTML, …) are converted to
    # Markdown by a `docling-serve` container (the preferred, AI-grade path).
    # When it's disabled or unreachable, ingestion falls back to the in-process
    # MarkItDown parser so uploads never hard-fail. Plain text/markdown is read
    # directly and never touches either.
    docling_enabled: bool = True
    docling_url: str = "http://docling:5001"  # empty disables → MarkItDown fallback
    docling_timeout_s: int = 120  # per-file convert timeout (heavy PDFs are slow)
    # Upload guardrails (also bound zip-bomb blast radius).
    ingest_max_file_mb: int = 25  # reject a single member larger than this
    ingest_max_files: int = 200  # max files (post zip-expansion) per request
    ingest_max_archive_uncompressed_mb: int = 200  # total unzipped size cap (per zip)
    ingest_max_request_mb: int = 500  # total bytes processed per upload request
    # Uploaded files and manually-provided context are persisted as artifacts on
    # disk (a mounted volume), organized into per-user/project/category
    # subfolders. Each produced memory's source_ref then references the stored
    # artifact (path + a /v1/ingest/artifacts/{id} download handle) so it's
    # traceable and re-fetchable. The API and ingest worker MUST share this
    # volume (RWX) — the API writes the file, the worker reads it back to parse.
    # (Object storage — GCS/S3 — is a later swap behind this same interface.)
    ingest_storage_enabled: bool = True
    ingest_storage_dir: str = "~/.neuralscape/ingest"  # volume mount in prod

    # ── Visual setup exemplars (VISUAL_EXEMPLARS_SPEC) ───────────────
    # The trading adapter can extract chart images from a book, store the bytes,
    # vision-describe each with a multimodal model, and index the description as a
    # normal memory (category `setup`, tag `visual_exemplar`) + a VisualExemplar
    # graph node. Dark by default. v1 storage = a local dir addressed by a
    # file:// URI (source_ref.stored_path); swap to S3/MinIO later behind the
    # same stored_path interface by changing only exemplar_store helpers.
    exemplar_store_enabled: bool = False
    exemplar_store_dir: str = "~/.neuralscape/exemplars"  # object store (local, v1)
    # Request figure/picture extraction from docling-serve so PDF images are
    # available to the exemplar pipeline (adds latency; only needed with
    # exemplar_store_enabled).
    docling_extract_images: bool = False
    # Multimodal model that describes a setup image. Routed through the LLM
    # gateway (llm_gateway_*) which is what fronts Opus 4.8; empty ⇒ the gateway's
    # default llm model. Book-exemplar and live-chart reads must share this model
    # so their descriptions land in one visual vocabulary.
    exemplar_vision_model: str = ""

    # ── Code-graph adapter (Graphify, Phase F1) ──────────────────────
    # NS uses Graphify (PyPI: graphifyy, optional `code-graph` extra) as a
    # library over a `graph.json` code graph. The interaction surface is ALWAYS
    # Neuralscape: the query_code_graph / get_code_neighbors / code_path
    # MCP tools + /v1/code-graph/* REST routes delegate to the library — agents
    # never talk to Graphify's own MCP server.
    #
    # Default graph.json the code-graph query tools resolve against when the
    # caller doesn't pass a graph_id (an ingested bundle's artifact id). Empty ⇒
    # no default graph; tools then require graph_id. Per-project graphs need no
    # setting: each ingested graph.json is an owner-scoped artifact addressed by
    # its graph_id (stamped into every produced memory's source_ref).
    code_graph_json_path: str = ""
    # Confidence assigned per Graphify edge/insight confidence tag (F1 epistemic
    # mapping): EXTRACTED → epistemic_level="explicit", INFERRED → "deductive"
    # with reduced confidence, AMBIGUOUS → stored only when its assigned
    # confidence clears `code_graph_ambiguous_floor` — with the defaults below
    # (0.3 < 0.5) AMBIGUOUS-derived memories are DROPPED. Lower the floor (or
    # raise the ambiguous confidence) to keep them; kept ones are tagged
    # `ambiguous` for the dreaming sweep's contradiction pass.
    code_graph_extracted_confidence: float = 0.9
    code_graph_inferred_confidence: float = 0.6
    code_graph_ambiguous_confidence: float = 0.3
    code_graph_ambiguous_floor: float = 0.5
    # Caps for the semantic layer distilled from one graph.json (bound the blast
    # radius of a huge graph — we ingest the STABLE summary, never the raw graph).
    code_graph_max_communities: int = 50
    code_graph_max_god_nodes: int = 10
    code_graph_max_surprises: int = 10
    code_graph_max_rationale: int = 100

    # Auth
    # Legacy single shared API key. When set without `neuralscape_user_token_secret`,
    # all requests must present this exact token in `Authorization: Bearer ...` and
    # user_id is taken from the request body (trust-based, single-team or single-user).
    neuralscape_api_key: str = ""
    # Multi-user HMAC-signed tokens. When set, the server also accepts tokens of the
    # form `{base64url(payload)}.{hmac_sha256(secret, payload)}` where payload is
    # `{"user_id": "...", "exp": <unix-ts>}`. The verified user_id is attached to
    # `request.state.user_id` so routes don't need to trust the request body.
    # Generate tokens via `python scripts/issue_user_token.py --user <name>`.
    neuralscape_user_token_secret: str = ""
    # Public HTTPS base URL the service is reachable at from the internet
    # (e.g. the cloudflared tunnel hostname: "https://neuralscape.example.com").
    # Used as the OAuth issuer and to build the .well-known metadata URLs that
    # Claude Cowork / claude.ai fetch when connecting as a custom MCP connector.
    # Leave empty for local dev / Claude Code CLI (OAuth discovery is then off).
    # No trailing slash; a trailing slash is stripped at use sites.
    neuralscape_public_url: str = ""
    # OAuth access-token TTL (seconds). Anthropic silently refreshes via the
    # refresh token, so this can be short. Default 1 hour.
    oauth_access_ttl: int = 3600
    # OAuth refresh-token TTL (seconds). Default 30 days.
    oauth_refresh_ttl: int = 30 * 24 * 3600

    # ── Federated login providers (the consent-screen identity step) ──────
    # Selects HOW a human proves identity at the OAuth consent step. The MCP
    # OAuth machinery (DCR, auth codes, access/refresh tokens, BearerAuth) is
    # identical for every provider — only the /oauth/authorize login UX
    # changes. Pre-issued admin HMAC tokens keep working as Bearer credentials
    # regardless of this setting (so CLI/CI/e2e are unaffected).
    #   "token"    → paste an admin-issued HMAC token (default; legacy behavior)
    #   "google"   → Sign in with Google (OIDC), gated by the env allowlist
    #   "supabase" → Sign in via Supabase (Google), gated by Supabase's
    #                Before-User-Created hook (+ optional env allowlist)
    # This is a per-deployment switch: a managed deployment may set "supabase",
    # public installs set "google", local/dev leaves "token".
    auth_provider: str = "token"

    # When a federated provider (google/supabase) is active, the consent screen
    # ALSO offers an admin-issued-token paste box by default, so tokens from
    # issue_user_token.py keep working as a login option (service accounts, CI,
    # users without a Google identity). Set False for a provider-only login
    # screen (no token paste). Ignored when auth_provider="token" (the paste box
    # is the only mechanism there and always shown).
    auth_allow_token_paste: bool = True

    # Google OIDC (auth_provider="google"). Neuralscape is a *confidential*
    # client TO Google here (server-side code exchange) while still being a
    # *public* OAuth AS to the MCP client. Register this redirect URI in Google
    # Cloud → Credentials:  {NEURALSCAPE_PUBLIC_URL}/oauth/google/callback
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""

    # Email allowlist — the gate for "google"; an optional extra gate for
    # "supabase". A login is accepted when the verified email's domain is in
    # auth_allowed_domains OR the exact email is in auth_email_allowlist.
    # Both are comma-separated and case-insensitive. A leading "@" on a domain
    # is tolerated ("@example.com" == "example.com").
    auth_allowed_domains: str = ""   # e.g. "example.com"
    auth_email_allowlist: str = ""   # e.g. "alice@gmail.com,bob@x.com"

    # Supabase (auth_provider="supabase"). The browser sign-in page uses the
    # anon key; the returned session JWT is verified server-side. Provide
    # EITHER the legacy HS256 jwt secret, OR leave it empty to verify against
    # the project's asymmetric JWKS (auto-derived from supabase_url).
    supabase_url: str = ""           # e.g. "https://abcd.supabase.co" (no trailing slash)
    supabase_anon_key: str = ""      # publishable anon/publishable key (browser sign-in)
    supabase_jwt_secret: str = ""    # legacy HS256 secret; empty → use project JWKS

    # Identity mapping: verified email → existing Neuralscape user_id. Unmapped
    # emails are slugified deterministically (user_id can't contain "@"). Use
    # this to preserve memories already keyed by an existing id.
    # Format: "alice@example.com:alice,ops@example.com:ops-bot"
    auth_identity_map: str = ""

    # ── Dictator role + authoritative "standard" memory tier ─────────────
    # Feature-flagged OFF by default so the public/generic build behaves
    # identically until a deployment opts in.
    #   standards_enabled  → turns on the `standard` visibility tier: a
    #     dictator-only authoritative pool that every caller reads and that is
    #     always injected at session start (see the plugin + /v1/context).
    #   processes_enabled  → turns on the process registry (list_processes /
    #     get_process) built on top of `standard`-tier memories tagged
    #     `process:<slug>`.
    #   dictator_user_ids  → CSV allowlist of user_ids permitted to WRITE
    #     `standard`-tier memories (and to delete them). Everyone else can only
    #     read them. Empty means nobody can write standards.
    standards_enabled: bool = False
    processes_enabled: bool = False
    dictator_user_ids: str = ""   # e.g. "mark,alice"

    # Service
    host: str = "0.0.0.0"
    port: int = 8199
    default_user_id: str = "default_user"
    default_project_id: str | None = None
    mcp_transport: str = "stdio"  # "stdio" or "http"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("neuralscape_public_url")
    @classmethod
    def _validate_public_url(cls, value: str) -> str:
        """OAuth issuer must be a well-formed absolute URL, and HTTPS unless it
        points at a loopback host (local dev / OAuth testing). Empty disables
        OAuth discovery and is allowed. Trailing slash is normalized off."""
        if not value:
            return value
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        is_loopback = host in _LOOPBACK_HOSTS
        if not parsed.netloc or parsed.scheme not in ("https", "http"):
            raise ValueError("NEURALSCAPE_PUBLIC_URL must be an absolute http(s):// URL")
        if parsed.scheme != "https" and not is_loopback:
            raise ValueError("NEURALSCAPE_PUBLIC_URL must use https:// (except loopback hosts)")
        return value.rstrip("/")

    @field_validator("oauth_access_ttl", "oauth_refresh_ttl")
    @classmethod
    def _validate_positive_ttl(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("OAuth token TTLs must be > 0 seconds")
        return value

    @field_validator(
        "docling_timeout_s",
        "ingest_max_file_mb",
        "ingest_max_files",
        "ingest_max_archive_uncompressed_mb",
        "ingest_max_request_mb",
    )
    @classmethod
    def _validate_positive_ingest_limit(cls, value: int, info) -> int:
        # These feed the Docling timeout and the zip guardrails; a 0/negative
        # value would time out every conversion or invert the archive checks, and
        # only surface as a runtime failure after deploy. Reject at startup.
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return value

    @field_validator(
        "ask_timeout_minimal_s",
        "ask_timeout_low_s",
        "ask_timeout_medium_s",
        "ask_timeout_high_s",
        "queue_status_window_s",
    )
    @classmethod
    def _validate_positive_ask_limits(cls, value: int, info) -> int:
        # A 0/negative timeout would fail every answering call (or make the
        # queue-status window empty) and only surface at request time.
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return value

    @field_validator("auth_provider")
    @classmethod
    def _validate_auth_provider(cls, value: str) -> str:
        v = (value or "token").strip().lower()
        if v not in ("token", "google", "supabase"):
            raise ValueError(
                "AUTH_PROVIDER must be one of: token, google, supabase"
            )
        return v

    @field_validator("supabase_url")
    @classmethod
    def _normalize_supabase_url(cls, value: str) -> str:
        return (value or "").rstrip("/")

    # ── parsed views of the comma-separated auth settings ────────────────
    def allowed_domains_set(self) -> set[str]:
        """Lowercased allowed email domains, leading '@' stripped."""
        return {
            d.strip().lstrip("@").lower()
            for d in self.auth_allowed_domains.split(",")
            if d.strip()
        }

    def email_allowlist_set(self) -> set[str]:
        """Lowercased exact-match allowed emails."""
        return {
            e.strip().lower()
            for e in self.auth_email_allowlist.split(",")
            if e.strip()
        }

    def identity_map_dict(self) -> dict[str, str]:
        """Parse "email:user_id,email2:user_id2" into a lowercased-email map."""
        out: dict[str, str] = {}
        for pair in self.auth_identity_map.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            email, _, user_id = pair.partition(":")
            email = email.strip().lower()
            user_id = user_id.strip()
            if email and user_id:
                out[email] = user_id
        return out

    # ── dictator role helpers ────────────────────────────────────────────
    def dictator_user_ids_set(self) -> set[str]:
        """Parsed CSV of user_ids allowed to write `standard`-tier memories."""
        return {u.strip() for u in self.dictator_user_ids.split(",") if u.strip()}

    def is_dictator(self, user_id: str | None) -> bool:
        """True iff `user_id` is an authorized dictator (may write standards)."""
        return bool(user_id) and user_id in self.dictator_user_ids_set()

    def validate_required(self) -> None:
        """Validate that all required configuration fields are set.

        Raises:
            ValueError: If any required field is empty or missing.
        """
        errors = []
        # GOOGLE_API_KEY stays required even in gateway mode: graphiti's embedder
        # always runs on AI Studio (Vertex rejects batched embeds), and graphiti
        # defaults to AI Studio entirely unless LLM_GATEWAY_GRAPHITI_ENABLED.
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required but not set")
        if not self.neo4j_password:
            errors.append("NEO4J_PASSWORD is required but not set")
        if not self.neo4j_uri:
            errors.append("NEO4J_URI is required but not set")
        if not self.redis_url:
            errors.append("REDIS_URL is required but not set")
        if self.llm_gateway_enabled:
            if not self.llm_gateway_base_url:
                errors.append("LLM_GATEWAY_BASE_URL is required when LLM_GATEWAY_ENABLED is true")
            if not self.llm_gateway_api_key:
                errors.append("LLM_GATEWAY_API_KEY is required when LLM_GATEWAY_ENABLED is true")
        if self.connectors_enabled and not self.vault_key:
            errors.append("NEURALSCAPE_VAULT_KEY is required when CONNECTORS_ENABLED is true")
        # Federated login: a non-token provider needs the OAuth AS turned on
        # (public URL + signing secret) plus that provider's own credentials.
        if self.auth_provider in ("google", "supabase"):
            if not self.neuralscape_public_url:
                errors.append(
                    f"NEURALSCAPE_PUBLIC_URL is required when AUTH_PROVIDER={self.auth_provider}"
                )
            if not self.neuralscape_user_token_secret:
                errors.append(
                    f"NEURALSCAPE_USER_TOKEN_SECRET is required when AUTH_PROVIDER={self.auth_provider}"
                )
        if self.auth_provider == "google":
            if not self.google_oauth_client_id or not self.google_oauth_client_secret:
                errors.append(
                    "GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET are required "
                    "when AUTH_PROVIDER=google"
                )
            if not self.allowed_domains_set() and not self.email_allowlist_set():
                errors.append(
                    "AUTH_ALLOWED_DOMAINS or AUTH_EMAIL_ALLOWLIST must be set when "
                    "AUTH_PROVIDER=google (an empty allowlist denies everyone)"
                )
        if self.auth_provider == "supabase":
            if not self.supabase_url or not self.supabase_anon_key:
                errors.append(
                    "SUPABASE_URL and SUPABASE_ANON_KEY are required when AUTH_PROVIDER=supabase"
                )
        if errors:
            raise ValueError(
                "Missing required configuration:\n  - " + "\n  - ".join(errors)
            )

    def _gateway_openai_base(self) -> str:
        """OpenAI-compatible base URL for the gateway, normalized to end in /v1."""
        base = self.llm_gateway_base_url.rstrip("/")
        if base and not base.endswith("/v1"):
            base += "/v1"
        return base

    def _graphiti_ai_studio_models(self) -> dict:
        """graphiti (LLM + embedder + reranker) on Google AI Studio — clean,
        complete structured-output extraction. The default for graph writes."""
        return {
            "graphiti_llm_provider": "gemini",
            "graphiti_llm_model": self.gemini_llm_model,
            # Pin the small-model (cheap sub-tasks like edge dedup) to the main
            # model too — otherwise Graphiti's GeminiClient falls back to its
            # hardcoded DEFAULT_SMALL_MODEL, which is a flaky preview-tier model.
            "graphiti_llm_small_model": self.gemini_llm_model,
            "graphiti_llm_fallback_model": self.gemini_llm_fallback_model,
            "graphiti_llm_api_key": self.google_api_key,
            "graphiti_embedder_provider": "gemini",
            "graphiti_embedder_model": self.gemini_embedder_model,
            "graphiti_embedder_api_key": self.google_api_key,
            "graphiti_reranker_provider": "gemini",
        }

    def _graphiti_gateway_models(self) -> dict:
        """graphiti LLM + reranker through the gateway. Requires the NEURALSCAPE
        PATCH in mem0/mem0/memory/graphiti_memory.py (small_model threading +
        reranker config). Embedder stays on AI Studio: graphiti batches inputs
        and Vertex rejects multi-input embeds (400 batch_not_supported)."""
        return {
            "graphiti_llm_provider": "openai",
            "graphiti_llm_model": self.llm_gateway_graphiti_model,
            "graphiti_llm_small_model": self.llm_gateway_llm_model,
            "graphiti_llm_fallback_model": self.llm_gateway_llm_fallback_model,
            "graphiti_llm_api_key": self.llm_gateway_api_key,
            "graphiti_embedder_provider": "gemini",
            "graphiti_embedder_model": self.gemini_embedder_model,
            "graphiti_embedder_api_key": self.google_api_key,
            "graphiti_reranker_provider": "openai",
        }

    def get_mem0_config(self) -> dict:
        """Build mem0 config dict for Memory(config=...).

        The LLM + embedder route either through Google AI Studio (default) or,
        when ``llm_gateway_enabled``, an OpenAI-compatible gateway. The choice
        is a single env flag; everything else (model tags, keys, providers) is
        derived from it.
        """
        # Qdrant: server mode (url) or local on-disk mode (path)
        qdrant_config: dict = {
            "collection_name": self.qdrant_collection,
            "embedding_model_dims": 768,
        }
        if self.qdrant_url:
            qdrant_config["url"] = self.qdrant_url
        else:
            qdrant_config["path"] = str(Path(self.qdrant_path).expanduser())
            qdrant_config["on_disk"] = self.qdrant_on_disk

        if self.llm_gateway_enabled:
            gw = self._gateway_openai_base()
            # graphiti's OpenAI llm/embedder/reranker clients are built without
            # an explicit base_url by the mem0 adapter, so AsyncOpenAI falls
            # back to OPENAI_BASE_URL. Set it (and a default key) here, before
            # the graph clients are constructed, so the graph side also routes
            # through the gateway. mem0's own clients use openai_base_url below.
            # Set (not setdefault) so a stale OPENAI_API_KEY from the environment
            # can't shadow the gateway key for graphiti's env-fed openai clients.
            # base_url non-emptiness is guaranteed by validate_required() above.
            os.environ["OPENAI_BASE_URL"] = gw
            os.environ["OPENAI_API_KEY"] = self.llm_gateway_api_key

            llm_block = {
                "provider": "openai",
                "config": {
                    "model": self.llm_gateway_llm_model,
                    "api_key": self.llm_gateway_api_key,
                    "openai_base_url": gw,
                },
            }
            embedder_block = {
                "provider": "openai",
                "config": {
                    "model": self.llm_gateway_embedder_model,
                    "api_key": self.llm_gateway_api_key,
                    "openai_base_url": gw,
                    "embedding_dims": 768,
                },
            }
            # graphiti routes through the gateway only when explicitly enabled;
            # otherwise it stays on AI Studio for a clean graph (the gateway's
            # strict json_schema support for Gemini is incomplete → flaky/partial
            # extraction). The vector path above is what stabilizes the API.
            graphiti_models = (
                self._graphiti_gateway_models()
                if self.llm_gateway_graphiti_enabled
                else self._graphiti_ai_studio_models()
            )
        else:
            llm_block = {
                "provider": "gemini",
                "config": {
                    "model": self.gemini_llm_model,
                    "api_key": self.google_api_key,
                },
            }
            embedder_block = {
                "provider": "gemini",
                "config": {
                    "model": self.gemini_embedder_model,
                    "api_key": self.google_api_key,
                    "embedding_dims": 768,
                },
            }
            graphiti_models = self._graphiti_ai_studio_models()

        return {
            "llm": llm_block,
            "embedder": embedder_block,
            "vector_store": {
                "provider": "qdrant",
                "config": qdrant_config,
            },
            "graph_store": {
                "provider": "graphiti",
                "config": {
                    "url": self.neo4j_uri,
                    "username": self.neo4j_user,
                    "password": self.neo4j_password,
                    "database": self.neo4j_database,
                    **graphiti_models,
                    "store_raw_episode_content": self.store_raw_episode_content,
                    "update_communities": self.update_communities,
                },
            },
            "version": "v1.1",
        }


settings = Settings()


def parse_redis_settings() -> RedisSettings:
    """Parse redis_url into ARQ RedisSettings using urllib.parse for robustness.

    Handles formats: redis://host:port/db, redis://user:password@host:port/db
    """
    parsed = urlparse(settings.redis_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    database = int(parsed.path.lstrip("/")) if parsed.path and parsed.path.strip("/") else 0
    password = parsed.password

    return RedisSettings(
        host=host,
        port=port,
        database=database,
        password=password,
        conn_timeout=10,
        conn_retries=5,
        conn_retry_delay=2,
        retry_on_timeout=True,
    )
