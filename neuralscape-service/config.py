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
    gemini_llm_model: str = "gemini-3-flash-preview"
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

    # ── Data-layer connectors ─────────────────────────────────────────
    # When enabled, the service hosts connectors (Notion/Drive/MCP/REST),
    # stores their credentials encrypted in the vault, and runs a periodic
    # sync. `vault_key` is a Fernet key (generate with
    # `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
    connectors_enabled: bool = False
    vault_key: str = ""  # NEURALSCAPE_VAULT_KEY — required when connectors_enabled
    connector_sync_cron_hours: int = 6  # interval (hours) the sync cron fires

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
