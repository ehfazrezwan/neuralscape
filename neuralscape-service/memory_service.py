"""Business logic layer for neuralscape memory service.

Both REST endpoints and MCP tools call into this same MemoryService.

Facade over the ``memory/`` package (mechanical mixins-with-facade split):
the implementation lives in ``memory/*.py``; this module preserves the full
public + test import surface.
"""

import logging
import threading

# Kept as facade attributes: legacy patch points and callers reference
# ``memory_service.genai`` / ``memory_service.settings``.
from google import genai  # noqa: F401

from config import settings  # noqa: F401

from index_format import distill_title, estimate_tokens  # noqa: F401 — estimate_tokens re-exported for legacy callers/tests
# E2: write-time token stamping — a REAL tiktoken count when the savings
# meter is enabled, the len/4 heuristic when it's off (zero tokenizer calls).
# The stored field keeps its `token_estimate` name either way.
from savings_meter import stamp_tokens
from prompts import (
    build_extraction_messages,
    parse_extraction_response,
)
from schemas import (
    EPISTEMIC_LEVEL_VOCAB,
    GLOBAL_CATEGORIES,
    MEMORY_CATEGORIES,
    PROJECT_CATEGORIES,
    ContextResponse,
    MemoryResponse,
    MemoryScope,
    MemoryVisibility,
    default_scope_for_category,
    default_visibility_for_category,
    normalize_visibility,
    validate_occurred_at,
)

logger = logging.getLogger(__name__)

# ── Re-exports (public + test import surface; all verbatim moves) ──────
from memory.audit import _audit_log  # noqa: F401,E402
from memory.groups import (  # noqa: F401,E402
    _build_group_id,
    _check_edit_permission,
    _edge_is_invalidated,
    _get_group_ids,
    _live_edges_filter,
)
from memory.hashing import (  # noqa: F401,E402
    _created_at_key,
    _infer_project_id,
    _parse_expires_at,
    content_hash,
)
from memory.junk import (  # noqa: F401,E402
    _JUNK_PATTERNS,
    _JUNK_RE,
    _clean_conversation_for_graph,
    _deleted_msg,
    _is_junk_fact,
)
from memory.ranking import (  # noqa: F401,E402
    REINFORCEMENT_BOOST_K,
    REINFORCEMENT_TIMES_DERIVED_CAP,
    RRF_K,
    _dense_score_floor,
    _mem_is_tombstoned,
    _reinforcement_boost,
    _rrf_fuse,
    _salience_tiebreak,
    _times_derived_from_metadata,
    _unit_cosine,
)
from memory.reads import GET_MEMORIES_MAX_IDS, TIMELINE_MAX_DEPTH  # noqa: F401,E402
from memory.retry import _TRANSIENT_PATTERNS, _is_transient, retry_transient  # noqa: F401,E402
from memory.search import (  # noqa: F401,E402
    _GRAPH_SEARCH_JOIN_TIMEOUT_S,
    _GRAPH_SEARCH_POOL,
)
from memory.standards import _ALWAYS_INJECT_TAGS, _SLUG_RE  # noqa: F401,E402

from memory.convert import ConvertMixin  # noqa: E402
from memory.core import CoreMixin  # noqa: E402
from memory.delete import DeleteMixin  # noqa: E402
from memory.edit import EditMixin  # noqa: E402
from memory.graph_admin import GraphAdminMixin  # noqa: E402
from memory.reads import ReadsMixin  # noqa: E402
from memory.search import SearchMixin  # noqa: E402
from memory.standards import StandardsMixin  # noqa: E402
from memory.write import WriteMixin  # noqa: E402


class MemoryService(
    CoreMixin,
    ConvertMixin,
    WriteMixin,
    SearchMixin,
    ReadsMixin,
    StandardsMixin,
    EditMixin,
    DeleteMixin,
    GraphAdminMixin,
):
    """Encapsulates all memory business logic.

    Provides a unified interface used by both REST endpoints and MCP tools.
    """


# ──────────────────────────────────────────────
# Shared process-wide instance (audit 27 #35)
# ──────────────────────────────────────────────
#
# main.py (REST) and mcp_server.py (mounted MCP transport) run in the SAME
# process; each holding its own MemoryService used to double every lazy-init
# (two mem0/Graphiti stacks, two embedder clients) and split warm caches.
# Both now resolve the one instance below. Workers still construct their own
# service explicitly (separate processes, different lifecycle).

_shared_service: "MemoryService | None" = None
_shared_service_lock = threading.Lock()


def get_shared_service() -> "MemoryService":
    """Return the process-wide MemoryService, creating it on first use.

    Thread-safe double-checked locking; construction itself is cheap (all
    heavy clients inside MemoryService are lazy), so this only guarantees
    identity — REST and MCP share one instance and one set of warm caches.
    """
    global _shared_service
    if _shared_service is None:
        with _shared_service_lock:
            if _shared_service is None:
                _shared_service = MemoryService()
    return _shared_service
