"""Distill a Graphify ``graph.json`` into its STABLE semantic layer.

Hard rule (roadmap Phase F): the raw code graph is NEVER mirrored into
memories/Graphiti — it is huge and churns with every commit, and Graphify
already serves it live through NS's code-graph query tools. What gets ingested
is only the layer that stays true across commits:

- **communities** — LLM-labeled module purposes (``community_name`` on nodes);
- **god nodes** — the most-connected core abstractions (library-computed);
- **surprising connections** — non-obvious cross-module couplings
  (library-computed, each carrying its edge's confidence tag);
- **rationale nodes** — ``# NOTE:`` / ``# HACK:`` / docstring rationale the
  extractor lifted out of source (``file_type == "rationale"``).

Everything is computed **via the graphifyy library** (``god_nodes`` /
``surprising_connections`` are its public API; the loader and community
reconstruction are its serve-layer helpers) — NS-side code is only the mapping
onto memory categories + the confidence-tag → epistemic-level mapping:

    EXTRACTED  → epistemic_level="explicit"   (confidence: extracted)
    INFERRED   → epistemic_level="deductive"  (reduced confidence)
    AMBIGUOUS  → epistemic_level="deductive", stored ONLY when its assigned
                 confidence clears ``code_graph_ambiguous_floor`` (default:
                 dropped), tagged ``ambiguous`` for the dreaming sweep.

This module imports ``graphify`` at module level — import it only behind
:func:`adapters.code_graph.code_graph_available`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
from graphify.analyze import god_nodes as _god_nodes
from graphify.analyze import surprising_connections as _surprising_connections
from graphify.serve import _communities_from_graph, _load_graph

logger = logging.getLogger(__name__)


class CodeGraphError(Exception):
    """A graph.json could not be loaded/parsed."""


@dataclass(frozen=True)
class SemanticFact:
    """One semantic-layer memory candidate distilled from a code graph."""

    category: str  # module | hotspot | boundary | rationale
    content: str
    epistemic_level: str  # explicit | deductive | inductive
    confidence: float
    external_id: str  # node id / community id inside graph.json
    title: str
    tags: list[str] = field(default_factory=list)


def load_code_graph(path: str) -> nx.Graph:
    """Load a graph.json via the graphifyy library loader.

    ``graphify.serve._load_graph`` exits the process on a missing/corrupt file
    (it's written for the CLI); we convert that to :class:`CodeGraphError` so a
    bad path surfaces as an ingest/tool error — the same trick graphify's own
    MCP server uses internally (``_load_ctx``).
    """
    try:
        return _load_graph(path)
    except SystemExit as e:
        raise CodeGraphError(f"could not load graph.json at {path}") from e


def _confidence_mapping(tag: str, settings) -> tuple[str, float, list[str]] | None:
    """Map a Graphify confidence tag → (epistemic_level, confidence, tags).

    Returns ``None`` when the fact must be dropped (AMBIGUOUS below the floor).
    """
    tag = (tag or "EXTRACTED").strip().upper()
    if tag == "EXTRACTED":
        return "explicit", settings.code_graph_extracted_confidence, []
    if tag == "INFERRED":
        return "deductive", settings.code_graph_inferred_confidence, []
    # AMBIGUOUS: stored only above the configurable floor; default drop.
    conf = settings.code_graph_ambiguous_confidence
    if conf < settings.code_graph_ambiguous_floor:
        return None
    # Kept AMBIGUOUS facts are flagged for the dreaming sweep's contradiction pass.
    return "deductive", conf, ["ambiguous"]


def _community_facts(G: nx.Graph, settings) -> list[SemanticFact]:
    """One `module` fact per LLM-labeled community (module purposes)."""
    communities = _communities_from_graph(G)
    facts: list[SemanticFact] = []
    for cid in sorted(communities)[: settings.code_graph_max_communities]:
        members = communities[cid]
        name = None
        for nid in members:
            name = G.nodes[nid].get("community_name")
            if name:
                break
        if not name:
            # Unlabeled community — a bare cluster id is structure, not knowledge.
            continue
        labels = [
            str(G.nodes[n].get("label", n))
            for n in members
            # Rationale nodes are long comment strings — they get their own
            # `rationale` facts and would only bloat the member list here.
            if G.nodes[n].get("file_type") != "rationale"
        ]
        key_members = ", ".join(sorted(labels)[:8])
        facts.append(
            SemanticFact(
                category="module",
                content=(
                    f"Code module '{name}' (community {cid}) groups {len(members)} "
                    f"code-graph nodes. Key members: {key_members}."
                ),
                # The community label is Graphify's LLM generalizing over the
                # cluster — inductive knowledge, carried at reduced confidence.
                epistemic_level="inductive",
                confidence=settings.code_graph_inferred_confidence,
                external_id=f"community:{cid}",
                title=f"Module: {name}",
            )
        )
    return facts


def _god_node_facts(G: nx.Graph, settings) -> list[SemanticFact]:
    """One `hotspot` fact per god node (library-computed, structure-derived)."""
    facts: list[SemanticFact] = []
    for gn in _god_nodes(G, top_n=settings.code_graph_max_god_nodes):
        nid = gn["id"]
        data = G.nodes.get(nid, {})
        # The library ranks by degree without a floor; on a small graph that
        # would crown leaf nodes. A "hub" needs at least 3 connections, and
        # rationale nodes are comments, never abstractions.
        if data.get("file_type") == "rationale" or gn.get("degree", 0) < 3:
            continue
        src = data.get("source_file") or "unknown source"
        community = data.get("community_name")
        where = f" in module '{community}'" if community else ""
        facts.append(
            SemanticFact(
                category="hotspot",
                content=(
                    f"'{gn['label']}' is a god node of the codebase with "
                    f"{gn['degree']} connections{where} (defined in {src}) — a "
                    f"core abstraction many parts depend on; changes to it have "
                    f"wide blast radius."
                ),
                # Derived deterministically from EXTRACTED structure by the
                # library (degree ranking) → deductive at extracted confidence.
                epistemic_level="deductive",
                confidence=settings.code_graph_extracted_confidence,
                external_id=str(nid),
                title=f"Hotspot: {gn['label']}",
            )
        )
    return facts


def _surprise_facts(G: nx.Graph, settings) -> list[SemanticFact]:
    """One `boundary` fact per surprising cross-module connection.

    Each carries its underlying edge's confidence tag through the epistemic
    mapping — this is where the AMBIGUOUS floor applies.
    """
    communities = _communities_from_graph(G)
    facts: list[SemanticFact] = []
    surprises = _surprising_connections(
        G, communities, top_n=settings.code_graph_max_surprises
    )
    for i, s in enumerate(surprises):
        mapped = _confidence_mapping(s.get("confidence", "EXTRACTED"), settings)
        if mapped is None:
            logger.info(
                "Dropping AMBIGUOUS surprising connection %s -> %s "
                "(confidence %.2f below floor %.2f)",
                s.get("source"), s.get("target"),
                settings.code_graph_ambiguous_confidence,
                settings.code_graph_ambiguous_floor,
            )
            continue
        epistemic, conf, tags = mapped
        rel = s.get("relation") or "connection"
        why = s.get("why") or "cross-module semantic connection"
        files = [f for f in (s.get("source_files") or []) if f]
        via = f" ({files[0]} ↔ {files[1]})" if len(files) == 2 else ""
        facts.append(
            SemanticFact(
                category="boundary",
                content=(
                    f"Surprising cross-module connection: '{s.get('source')}' "
                    f"--{rel}--> '{s.get('target')}'{via}. Why it's non-obvious: {why}."
                ),
                epistemic_level=epistemic,
                confidence=conf,
                external_id=f"surprise:{i}:{s.get('source')}->{s.get('target')}",
                title=f"Boundary: {s.get('source')} -> {s.get('target')}",
                tags=tags,
            )
        )
    return facts


def _rationale_facts(G: nx.Graph, settings) -> list[SemanticFact]:
    """One `rationale` fact per rationale node (# NOTE:/# HACK:/docstrings).

    Rationale nodes are verbatim EXTRACTED source text → explicit.
    """
    facts: list[SemanticFact] = []
    count = 0
    for nid, data in G.nodes(data=True):
        if data.get("file_type") != "rationale":
            continue
        if count >= settings.code_graph_max_rationale:
            logger.info(
                "Rationale cap (%d) reached — remaining rationale nodes stay "
                "queryable via the code-graph tools",
                settings.code_graph_max_rationale,
            )
            break
        label = str(data.get("label", "")).strip()
        if not label:
            continue
        src = data.get("source_file") or "unknown source"
        loc = data.get("source_location") or ""
        where = f"{src} {loc}".strip()
        facts.append(
            SemanticFact(
                category="rationale",
                content=f'Code rationale ({where}): "{label}"',
                epistemic_level="explicit",
                confidence=settings.code_graph_extracted_confidence,
                external_id=str(nid),
                title=f"Rationale in {src}",
            )
        )
        count += 1
    return facts


def extract_semantic_layer(G: nx.Graph, settings) -> list[SemanticFact]:
    """The full stable semantic layer of one code graph, as memory candidates."""
    facts: list[SemanticFact] = []
    facts.extend(_community_facts(G, settings))
    facts.extend(_god_node_facts(G, settings))
    facts.extend(_surprise_facts(G, settings))
    facts.extend(_rationale_facts(G, settings))
    logger.info(
        "Code-graph semantic layer: %d facts (%d nodes / %d edges in graph)",
        len(facts), G.number_of_nodes(), G.number_of_edges(),
    )
    return facts
