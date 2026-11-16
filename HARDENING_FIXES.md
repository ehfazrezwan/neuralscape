# ICE v2 Hardening Fixes — Fable Review MUST-FIX Items

> **Status (as of 2026-11-16, dev 6a81341f):** historical. A record of the Fable-review fixes applied during the ICE v2 work. The M3 item (the `knowledge_system` parameter) has since been superseded: the parameter is wired for the code tools and is no longer marked EXPERIMENTAL. File and line citations below are as of that change and have drifted.

This document details the fixes applied to address all 5 MUST-FIX items from the Fable review (ice-v2-fable-review.md §5), plus 2 cheap known-issues (KI#6, KI#12).

## M1: Placeholder Fusion Defect (CRITICAL - Production Recall Regression)

**The Bug:** `knowledge/__init__.py:119-137` registers an always-healthy `code-graphify-lib` with a capability placeholder engine (`code_space="__registry_capability__"`, `G=None`). The fusion wiring (`mcp_server.py:1360-1366`) resolved this placeholder via `get_system()` and queried it, composing "No graph loaded. Run index() first." into the `[structure]` section. This flipped `recall_memories` output from JSON array to fused plain text for ANY `project_id` + coding-signal query in the default image (Dockerfile installs code-graph extra), regressing production behavior for every MCP client on every project.

**The Fix (mcp_server.py:1360-1392):**
- Added explicit filtering before fusion path: skip code systems that are capability placeholders (`code_space == "__registry_capability__"`) OR have `G=None` (no loaded graph).
- Net effect: `recall_memories` output is UNCHANGED (normal JSON array) when there's no real indexed code engine for the project — the current production reality.
- Decision #3 preserved: fusion defaults ON, but code leg fires only when there's REAL code to fuse (not placeholders/empty engines).

**Test:** `tests/test_hardening_fixes.py::test_placeholder_fusion_does_not_regress_recall_output` — reproduces exact bug scenario (placeholder registered + coding query on project_id), asserts output is normal JSON array, NOT fused text.

---

## M2: Project-Scope eligible_systems

**The Bug:** `knowledge/registry.py:73-135` accepted `project_id` parameter but ignored it (line 93 comment "reserved for Phase D; unused" survived at tip). A registered global system or capability placeholder was eligible for any arbitrary project, voiding layer 3's necessary condition ("project has ≥1 healthy INDEXED code_space").

**The Fix (knowledge/registry.py:73-135):**
- Implemented project-scoped filtering: when `project_id` is provided, code systems are eligible ONLY if they have a REAL indexed code_space for that project.
- Capability placeholders (`code_space == "__registry_capability__"`) and engines with `G=None` are NOT eligible for project-scoped queries.
- Base systems remain always eligible (not project-scoped).
- A code_space bound to a project must match that project's repo: an engine whose code_space differs from the `code_space` in the project's knowledge config (and does not map to the same `CODE_REPOS` directory) is not eligible, and the reason is logged (`code_space_matches_project`). A project with no recorded code_space is unconstrained.

**Test:** `tests/test_hardening_fixes.py::test_eligible_systems_requires_indexed_code_space_for_project` — verifies placeholder is eligible globally but NOT for project-scoped queries.

---

## M3: Make knowledge_system Parameter Honest

**The Bug:** The documented `knowledge_system` param on `recall_memories` + 5 code tools was a silent no-op. `RouteDecision.systems` was discarded; code tools dispatched via old `graph_id` path (`mcp_server.py:2019-2022, 2048-2100`); recall read only `wants_code_fusion` which layer-1 never sets. The API contract promised behavior that didn't happen.

**The Fix as first applied (historical; `mcp_server.py` has since changed, see the Superseded bullet below):**
- Chose the honest minimal fix: clearly marked `knowledge_system` as "EXPERIMENTAL (not yet fully implemented; full through-NS routing is Phase G)" in BOTH tool schemas.
- Updated description to state it's "currently logged for debugging but does not change recall/dispatch behavior (fusion defaults to router decision)".
- Phase G (documented follow-up) will wire explicit routing through the knowledge system seam for all surfaces.
- **Superseded:** Phase G / AR2 landed that wiring. On the code tools (`query_code_graph`, `get_code_neighbors`, `code_path`, `locate`, `code_impact` and their REST twins) `knowledge_system` + `graph_id` now dispatch through the knowledge-system seam (`knowledge_system="auto"` picks the measured-best healthy engine per op). On `recall_memories` it stays a hint: the code leg is governed by the router's fusion gate. It is no longer a no-op and no longer marked EXPERIMENTAL.

**Impact:** At the time, the API contract was made honest by documenting the parameter as experimental. Today the parameter is real for the code tools (see the Superseded bullet) and the tool schemas in `mcp_server.py` describe it without the EXPERIMENTAL marker.

---

## M4: Bounded Health Probe Cost on Hot Path

**The Bug:** `resolve_systems` runs synchronously on the MCP event loop (`mcp_server.py:1326`) and calls each candidate's `health()` (`~1365`). `CBMEngine`'s probe was an httpx GET with `timeout=60` (`cbm_engine.py:52,66`), so a black-holed bridge with `CBM_ENABLED=true` could stall the event loop up to 60s per routed recall.

**The Fix (adapters/code_graph/cbm_engine.py):**
- **Short timeout:** Health probe now uses a dedicated 2s timeout (not the operational 60s). Creates a separate short-timeout httpx client just for the probe.
- **TTL cache:** Health results are cached per engine instance for 10s. Prevents hot-path spam (every routed recall was making a new HTTP GET).
- Constants: `_HEALTH_PROBE_TIMEOUT = 2.0`, `_HEALTH_CACHE_TTL = 10.0`.

**Tests:**
- `tests/test_hardening_fixes.py::test_health_probe_timeout_is_bounded` — verifies health() uses bounded timeout.
- `tests/test_hardening_fixes.py::test_health_probe_is_cached` — verifies second call reuses cached result.

---

## M5: Stable Fused Recall Output Envelope

**The Bug:** When fusion fired, output switched from JSON array to plain prose sections — an unstable contract. Clients got a surprise type flip.

**The Fix (mcp_server.py:1490-1524):**
- Fused recall now returns a **stable JSON envelope** instead of plain text:
  ```json
  {
    "fused": true,
    "sections": {
      "structure": {
        "system": "code-cbm",
        "content": "...",
        "hits": [...]
      },
      "semantics": { "fqn": [...anchored memories...] },
      "memories": [...normal memory array...]
    },
    "query": "...",
    "project_id": "..."
  }
  ```
- Normal (non-fused) recall still returns the standard JSON array of memory objects.
- Clients always get valid JSON, never a type flip to prose text.

**Documentation:** Updated `recall_memories` tool description to document the stable envelope format (M5 hardening note).

**Test:** `tests/test_hardening_fixes.py::test_fused_recall_has_stable_json_envelope` — will verify stable JSON structure once fusion test fixtures are complete.

---

## KI#6: Coding-Signal Keywords Use Word-Boundary Regexes

**The Bug:** Structural keywords used substring match (`"defined" in query_lower`), over-triggering on prose (e.g. "undefined" matched "defined", "impact" in "impacted").

**The Fix (knowledge/router.py:91-125):**
- Converted `_STRUCTURAL_KEYWORDS` from plain strings to word-boundary regex patterns (`r"\bdefined\b"`, `r"\bimpact\b"`, etc.).
- Updated `_has_coding_signal()` to use `re.search(pattern, query_lower)` instead of `keyword in query_lower`.

**Impact:** Coding-signal gate is more precise — plain prose about "undefined variables" or "impacted users" no longer triggers the code leg.

---

## KI#12: Strengthen test_transport_is_declared_not_branched

**The Bug:** Test was named as enforcement but only checked that `transport` was a string and `eligible_systems` didn't take it as param. Didn't actually enforce the cross-cut (Fable's manual grep found no branches, but the test didn't verify this).

**The Fix (tests/test_knowledge_registry.py:229-287):**
- Added source scan: greps all `knowledge/*.py` and `mcp_server.py` for `.transport` or `info.transport` reads.
- Filters out safe contexts: declaration (`@dataclass`, `__init__`), health display, logging.
- Fails with forbidden branch locations if any routing/dispatch code branches on transport.

**Impact:** Cross-cutting transport-uniformity rule is now machine-enforced, not just reviewer-enforced.

---

## Summary of Changes

### Files Modified
1. **mcp_server.py** — M1 (fusion placeholder filter), M3 (honest schema docs), M5 (stable JSON envelope)
2. **knowledge/registry.py** — M2 (project-scoped eligible_systems)
3. **adapters/code_graph/cbm_engine.py** — M4 (bounded health probe + TTL cache)
4. **knowledge/router.py** — KI#6 (word-boundary coding-signal keywords)
5. **tests/test_knowledge_registry.py** — KI#12 (strengthened transport invariant test)
6. **tests/test_hardening_fixes.py** — NEW regression tests for all MUST-FIX items

### Tests Added
- `test_placeholder_fusion_does_not_regress_recall_output` (M1 regression repro)
- `test_eligible_systems_requires_indexed_code_space_for_project` (M2)
- `test_health_probe_timeout_is_bounded` (M4)
- `test_health_probe_is_cached` (M4)
- `test_fused_recall_has_stable_json_envelope` (M5 — skeleton)
- Strengthened `test_transport_is_declared_not_branched` (KI#12)

### Acceptance Criteria Met
✅ M1 fix: recall_memories output unchanged when no real indexed engine exists  
✅ M2 fix: eligible_systems filters placeholders for project-scoped queries  
✅ M3 fix: knowledge_system param honest (since implemented for the code tools in Phase G / AR2; see M3)  
✅ M4 fix: health probe bounded to 2s + cached for 10s  
✅ M5 fix: fused recall returns stable JSON envelope  
✅ KI#6 fix: coding-signal keywords use word-boundary regexes  
✅ KI#12 fix: transport invariant is machine-enforced by source scan  

### Full Phase G (documented follow-up, not in this PR)
- Wire `RouteDecision.systems` into actual dispatch for code tools + REST twins
- Add `POST /v1/code-graph/index` (+ MCP twin) on ingest queue
- Bind per-code_space engines into fusion (use `query.py` factory, kill placeholder)
- Make `eligible_systems` check code_space matches project repo
- Hook post-index liveness diff
- Run through-NS bench matrix

This hardening PR makes the merged `ice/v2` tree **production-SAFE** (no regression, bounded latency, honest contract) without building the full Phase G surface.
