"""CodeKnowledgeSystem — wraps CodeIntelEngine implementations as KnowledgeSystems.

One registry entry per backend (code-graphify-json for the artifact path; code-native
for the frozen NativeEngine if CODE_NATIVE_ENABLED=true). CBMEngine and GraphifyLibEngine
are Phases C and F — not built here.

The wrapper maps op-class names (query, neighbors, path, locate, impact) to the
engine's protocol methods and declares only capabilities the backend genuinely
supports (N/A honesty). ``transport`` reflects the engine's implementation
(in-process for these).
"""

from __future__ import annotations

import logging

from knowledge.base import (
    HealthStatus,
    IndexReport,
    IndexRequest,
    KnowledgeSystemInfo,
    RecallRequest,
    SystemAnswer,
    TaskRef,
)

logger = logging.getLogger(__name__)


class CodeKnowledgeSystem:
    """Wrap a CodeIntelEngine as a KnowledgeSystem (one entry per backend).

    Maps RecallRequest.operation → engine protocol methods. Declares only
    capabilities the backend supports (GraphifyJsonEngine: query/neighbors/path;
    NativeEngine: all ops).
    """

    def __init__(
        self,
        name: str,
        engine,  # CodeIntelEngine protocol (graphify_engine.py, native_engine.py)
        capabilities: frozenset[str],
        transport: str = "in-process",
        version: str | None = None,
    ):
        """Initialize a code knowledge system wrapper.

        Args:
            name: Registry key (e.g. "code-graphify-json", "code-native").
            engine: A CodeIntelEngine implementation (GraphifyJsonEngine, NativeEngine).
            capabilities: Op-class names this backend supports (honest N/A).
            transport: Implementation transport ("in-process" for these; "http" for CBM in Phase C).
            version: Engine version stamp (for bench attribution; None = unstamped).
        """
        self.info = KnowledgeSystemInfo(
            name=name,
            kind="code",
            capabilities=capabilities,
            transport=transport,
        )
        self._engine = engine
        self._version = version

    def health(self) -> HealthStatus:
        """Health check: is the engine reachable and ready?

        For engines that own a reachability probe (``engine.health()`` — e.g.
        CBMEngine, which pings the bridge's /health), that probe is authoritative:
        a DOWN bridge makes this system report ``unreachable`` and become
        INELIGIBLE for routing (PLAN §3.3 — a down system is never an error to
        recall; the base always answers).

        For in-process engines with no probe (GraphifyJsonEngine, NativeEngine),
        health is the cheap check: the engine object (and its loaded graph) exists.
        """
        try:
            # http/service engines expose a real reachability probe. Prefer it so
            # a DOWN CBM bridge makes code-cbm ineligible instead of falsely "ok".
            engine_health = getattr(self._engine, "health", None)
            if callable(engine_health):
                ok = engine_health()
                if ok:
                    return HealthStatus(
                        status="ok",
                        details={"engine_type": type(self._engine).__name__, "probe": True},
                    )
                return HealthStatus(
                    status="unreachable",
                    details={
                        "engine_type": type(self._engine).__name__,
                        "probe": True,
                        "reason": "engine health probe not-ok (e.g. bridge unreachable)",
                    },
                )

            # In-process engines: check that the engine's core state exists.
            # GraphifyJsonEngine: has .G (NetworkX graph).
            # NativeEngine: has ._graph_qdrant / ._code_neo4j (checked at init).
            if hasattr(self._engine, "G"):
                # GraphifyJsonEngine
                if self._engine.G is not None:
                    return HealthStatus(
                        status="ok",
                        details={"graph_loaded": True, "nodes": self._engine.G.number_of_nodes()},
                    )
                else:
                    return HealthStatus(
                        status="degraded",
                        details={"graph_loaded": False},
                    )
            # Assume healthy if engine object exists (NativeEngine lazy-inits its stores).
            return HealthStatus(
                status="ok",
                details={"engine_type": type(self._engine).__name__},
            )
        except Exception as e:
            logger.exception("Code system %s health check failed", self.info.name)
            return HealthStatus(
                status="unreachable",
                details={"error": str(e)},
            )

    def recall(self, req: RecallRequest) -> SystemAnswer:
        """Delegate to the engine's protocol methods based on operation hint.

        Maps RecallRequest.operation → CodeIntelEngine methods:
          - "query" → engine.query(question, mode, depth, token_budget)
          - "neighbors" → engine.neighbors(label, relation_filter)
          - "path" → engine.path(source, target, max_hops)
          - "locate" → engine.locate(query, k)
          - "impact" → engine.detect_changes() (Phase E)

        Raises EngineCapabilityError if the operation isn't in the system's
        declared capabilities (honest N/A).
        """
        op = req.operation or "query"  # Default to query if not specified

        # Capability check: fail fast if the backend doesn't support this op.
        if op not in self.info.capabilities:
            from adapters.code_graph.engine import EngineCapabilityError

            raise EngineCapabilityError(
                f"System {self.info.name} does not support operation '{op}' "
                f"(capabilities: {sorted(self.info.capabilities)})"
            )

        # Dispatch to the engine method.
        try:
            if op == "query":
                content = self._engine.query(
                    question=req.query,
                    mode=req.mode or "bfs",
                    depth=req.depth or 3,
                    token_budget=req.token_budget or 2000,
                )
                return SystemAnswer(
                    system_name=self.info.name,
                    system_version=self._version,
                    content=content,
                    hits=None,  # Text-only rendering for query
                    metadata={"operation": op},
                )

            elif op == "neighbors":
                if not req.label:
                    raise ValueError("neighbors operation requires 'label' parameter")
                content = self._engine.neighbors(
                    label=req.label,
                    relation_filter=req.relation_filter or "",
                )
                return SystemAnswer(
                    system_name=self.info.name,
                    system_version=self._version,
                    content=content,
                    hits=None,
                    metadata={"operation": op, "label": req.label},
                )

            elif op == "path":
                if not req.source or not req.target:
                    raise ValueError("path operation requires 'source' and 'target' parameters")
                content = self._engine.path(
                    source=req.source,
                    target=req.target,
                    max_hops=req.max_hops or 8,
                )
                return SystemAnswer(
                    system_name=self.info.name,
                    system_version=self._version,
                    content=content,
                    hits=None,
                    metadata={"operation": op, "source": req.source, "target": req.target},
                )

            elif op == "impact":
                # impact = blast radius: detect_changes(seed) → ChangeReport.
                # Seed symbol comes from label/source/query (first non-empty).
                # This is Phase F's "affected powers impact" deliverable.
                seed = req.label or req.source or req.query
                report = self._engine.detect_changes(since=seed)
                # ChangeReport is a dataclass — surface the blast-radius symbols
                # (modified_symbols) as structured hits + a text rendering.
                affected = list(getattr(report, "modified_symbols", []) or [])
                deleted = list(getattr(report, "deleted_symbols", []) or [])
                added = list(getattr(report, "added_symbols", []) or [])
                summary = getattr(report, "summary", "") or ""
                hits = [{"fqn": fqn, "impact": "affected"} for fqn in affected]
                hits += [{"fqn": fqn, "impact": "deleted"} for fqn in deleted]
                hits += [{"fqn": fqn, "impact": "added"} for fqn in added]
                lines = [summary] if summary else []
                lines += [f"  ~ {fqn}" for fqn in affected]
                lines += [f"  - {fqn}" for fqn in deleted]
                lines += [f"  + {fqn}" for fqn in added]
                content = "\n".join(lines) if lines else "No impact detected."
                return SystemAnswer(
                    system_name=self.info.name,
                    system_version=self._version,
                    content=content,
                    hits=hits or None,
                    metadata={
                        "operation": op,
                        "seed": seed,
                        "affected_count": len(affected),
                    },
                )

            elif op == "locate":
                # locate returns list[LocateHit]; convert to structured hits.
                hits_obj = self._engine.locate(query=req.query, k=req.limit)
                hits = [
                    {
                        "fqn": h.fqn,
                        "kind": h.kind,
                        "file": h.file,
                        "line": h.line,
                        "signature": h.signature,
                        "docstring": h.docstring,
                        "score": h.score,
                    }
                    for h in hits_obj
                ]
                # Render as text too (for backward compat with tools that expect content).
                lines = [f"{i+1}. {h['fqn']} ({h['kind']}) — {h['file']}:{h['line']}" for i, h in enumerate(hits)]
                content = "\n".join(lines) if lines else "No symbols found."
                return SystemAnswer(
                    system_name=self.info.name,
                    system_version=self._version,
                    content=content,
                    hits=hits,
                    metadata={"operation": op, "result_count": len(hits)},
                )

            else:
                # Shouldn't reach here (capability check above), but be defensive.
                raise ValueError(f"Unknown operation: {op}")

        except Exception as e:
            logger.exception("Code system %s recall failed for operation %s", self.info.name, op)
            # Re-raise so the caller sees the error (don't swallow engine failures).
            raise

    def index(self, req: IndexRequest) -> TaskRef | IndexReport:
        """Trigger indexing via the engine's index() method.

        GraphifyJsonEngine raises EngineCapabilityError (immutable artifact).
        NativeEngine (Phase A fixes) returns IndexReport.
        CBM (Phase C) will enqueue to ARQ and return TaskRef.
        """
        from adapters.code_graph.engine import EngineCapabilityError

        if "index" not in self.info.capabilities:
            raise EngineCapabilityError(
                f"System {self.info.name} does not support indexing "
                "(artifact-based engines are read-only)"
            )

        # Delegate to engine.index() — NativeEngine in Phase A, CBM in Phase C.
        report = self._engine.index(source=req.source, incremental=req.incremental)
        # Stamp version if available.
        if self._version and hasattr(report, "system_version"):
            report.system_version = self._version
        return report
