"""mem0/Qdrant result-to-response conversion helpers.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

from config import settings
from schemas import MemoryResponse, MemoryVisibility

class ConvertMixin:
    """ConvertMixin for MemoryService (mechanical split — see memory_service.py)."""

    # ──────────────────────────────────────────────
    # Retrieval economics (C1 batch get + C2 timeline)
    # ──────────────────────────────────────────────

    @staticmethod
    def _payload_readable_by(payload: dict, caller_user_id: str | None) -> bool:
        """Whether ``caller_user_id`` may read a Qdrant point's payload.

        Mirrors the pool rules of ``search()``: you can always read what you
        own (top-level ``user_id`` namespace or ``metadata.owner_user_id``);
        everyone reads ``shared``; everyone reads ``standard`` when the tier
        is enabled. Legacy rows without a visibility field are de-facto
        private to their writer.
        """
        if caller_user_id and payload.get("user_id") == caller_user_id:
            return True
        meta = payload.get("metadata") or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        if caller_user_id and meta.get("owner_user_id") == caller_user_id:
            return True
        vis = meta.get("visibility")
        if vis == MemoryVisibility.SHARED.value:
            return True
        if vis == MemoryVisibility.STANDARD.value and settings.standards_enabled:
            return True
        return False

    def _point_to_response(self, point) -> MemoryResponse:
        """Convert a raw Qdrant point (retrieve/scroll result) to a MemoryResponse."""
        payload = getattr(point, "payload", None) or {}
        return self._mem_to_response(
            {
                "id": str(getattr(point, "id", "")),
                "memory": payload.get("data", ""),
                "metadata": payload.get("metadata", {}) or {},
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
            }
        )

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _extract_memory_list(self, result) -> list[dict]:
        """Extract the list of memories from a mem0 result (handles both v1.0 and v1.1 formats)."""
        if isinstance(result, dict):
            return result.get("results", [])
        if isinstance(result, list):
            return result
        return []

    def _mem_to_response(self, mem: dict) -> MemoryResponse:
        """Convert a mem0 memory dict to a MemoryResponse.

        mem0's _search_vector_store / _get_all_from_vector_store helpers lift
        every non-promoted payload field into a top-level "metadata" dict.
        Because our Qdrant payload already nests our fields under a literal
        "metadata" key, the returned shape is {"metadata": {"metadata": {...}}}.
        Unwrap one level if we see that pattern so category/scope/project_id/
        tags/source resolve to their real values.

        Memory-model v2 fields (domain, observation_type, concepts, source_type,
        related_memory_ids, confidence, expires_at) surface as nulls for legacy
        memories that didn't store them — no migration needed.
        """
        metadata = mem.get("metadata", {}) or {}
        if isinstance(metadata.get("metadata"), dict):
            metadata = metadata["metadata"]
        return MemoryResponse(
            id=mem.get("id", ""),
            memory=mem.get("memory", ""),
            category=metadata.get("category"),
            scope=metadata.get("scope"),
            project_id=metadata.get("project_id"),
            tags=metadata.get("tags"),
            score=mem.get("score"),
            created_at=mem.get("created_at"),
            updated_at=mem.get("updated_at"),
            occurred_at=metadata.get("occurred_at"),
            source="vector",
            domain=metadata.get("domain"),
            observation_type=metadata.get("observation_type"),
            concepts=metadata.get("concepts"),
            source_type=metadata.get("source_type"),
            related_memory_ids=metadata.get("related_memory_ids"),
            confidence=metadata.get("confidence"),
            expires_at=metadata.get("expires_at"),
            derived_from=metadata.get("derived_from"),
            epistemic_level=metadata.get("epistemic_level"),
            memory_kind=metadata.get("memory_kind"),
            source_ref=metadata.get("source_ref"),
            visibility=metadata.get("visibility"),
            owner_user_id=metadata.get("owner_user_id"),
            title=metadata.get("title"),
            token_estimate=metadata.get("token_estimate"),
            speaker=metadata.get("speaker"),
        )

    def _result_to_responses(
        self, result, category: str | None = None, scope: str | None = None
    ) -> list[MemoryResponse]:
        """Convert a mem0 add() result to MemoryResponse list."""
        memories = self._extract_memory_list(result)
        responses = []
        for mem in memories:
            resp = self._mem_to_response(mem)
            if category and not resp.category:
                resp.category = category
            if scope and not resp.scope:
                resp.scope = scope
            responses.append(resp)
        return responses

    def _results_to_responses(self, results) -> list[MemoryResponse]:
        """Convert mem0 search/get_all results to MemoryResponse list."""
        memories = self._extract_memory_list(results)
        return [self._mem_to_response(mem) for mem in memories]

    @staticmethod
    def normalize_memory_content(text: str) -> str:
        """Normalize memory content for cross-source duplicate detection."""
        return (text or "").strip().lower()

    @staticmethod
    def find_duplicate_content(normalized: str, candidates) -> str | None:
        """Return the first candidate that content-matches ``normalized``.

        A match is exact equality or a substring in either direction — the
        same rule ``_deduplicate_responses`` uses to collapse a Graphiti
        edge against its Qdrant twin. Shared with the ask path (audit 27
        #17) so both dedup passes agree on what "the same fact" means.
        """
        for other in candidates:
            if normalized == other or normalized in other or other in normalized:
                return other
        return None
