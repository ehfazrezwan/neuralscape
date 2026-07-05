"""NS schema extensions for the embedded Kuzu graph (solo engine, unit 3 Tier 0).

Graphiti's KuzuDriver declares a STATIC schema (Entity/Episodic/Community/
RelatesToNode_/Saga + rel tables). NS stamps back-reference and enrichment
properties onto graph nodes — memory_id, wiki_path, dream_*, ns_* — and links
provenance ``Source`` nodes via ``DERIVED_FROM``; none of that exists in the
base schema, and Kuzu rejects ``SET n.<undeclared>`` and unknown labels at
bind time. This module declares the missing surface.

Applied once per driver construction (memory/core.py, kuzu provider only),
after ``KuzuDriver.setup_schema()``. Neo4j is schema-free and never needs it.
See docs/neuralscape/29-kuzu-port-inventory.md.
"""

import logging

logger = logging.getLogger(__name__)

# NS columns stamped onto graphiti-owned tables. All STRING deliberately: on
# schema-free Neo4j these arrive as plain strings (ISO text for the *_at
# fields) and NS only stores/reads them back — no temporal comparisons — so
# Kuzu mirrors that rather than inventing a stricter type contract.
_NS_NODE_COLUMNS: list[tuple[str, str]] = [
    ("memory_id", "STRING"),
    ("ns_visibility", "STRING"),
    ("ns_owner", "STRING"),
    ("ns_connector_id", "STRING"),
    ("ns_connector_type", "STRING"),
    ("ns_source_url", "STRING"),
    ("wiki_path", "STRING"),
    ("wiki_synthesized_at", "STRING"),
    ("strategy_playbook_path", "STRING"),
    ("strategy_synthesized_at", "STRING"),
    ("dream_superseded_by", "STRING"),
    ("dream_invalidated_at", "STRING"),
    ("dream_path", "STRING"),
    ("dreamt_at", "STRING"),
]

# RelatesToNode_ is included because Kuzu reifies RELATES_TO edges through it —
# edge-level NS props (memory_id, wiki_path) land on that node table. Saga is
# included because Neo4j's label-less `MATCH (n)` stamps Saga nodes too, and
# the per-table Kuzu stamp loops must be able to mirror that.
_NS_TABLES = ("Entity", "Episodic", "Community", "RelatesToNode_", "Saga")

# Provenance surface used by extensions/dreaming/graph_patcher.attach_source_ref.
# Kuzu node tables need a single-column primary key; the Kuzu branch of
# attach_source_ref keys Source rows on `key` = "<connector_id>::<source_key>"
# (Neo4j MERGEs on the two-column pair instead).
_NS_TABLE_DDL = [
    """CREATE NODE TABLE IF NOT EXISTS Source (
        key STRING PRIMARY KEY,
        connector_id STRING,
        source_key STRING,
        connector_type STRING,
        url STRING,
        title STRING,
        external_id STRING,
        last_synced_at STRING
    )""",
    """CREATE REL TABLE IF NOT EXISTS DERIVED_FROM(
        FROM Entity TO Source,
        FROM Episodic TO Source,
        FROM Community TO Source,
        FROM Saga TO Source
    )""",
]


# Kuzu ships FTS as a bundled extension (0.11.3 bundled all extensions in the
# farewell release) — it must be installed+loaded per Database before the FTS
# indices graphiti's BM25 search leg expects can exist. Nothing in the subtree
# driver creates them (build_indices_and_constraints is a Kuzu no-op), so this
# bootstrap also runs graphiti's own get_fulltext_indices(KUZU) statements —
# fixing graphiti's hybrid search on Kuzu, not just NS's queries.
_FTS_BOOTSTRAP = ["INSTALL FTS", "LOAD EXTENSION FTS"]


def _graphiti_fts_statements() -> list[str]:
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.graph_queries import get_fulltext_indices

    return get_fulltext_indices(GraphProvider.KUZU)


def ns_kuzu_schema_statements() -> list[str]:
    """All bootstrap statements, ordered: FTS extension → tables/columns →
    FTS indices (which index columns that must exist first)."""
    stmts = list(_FTS_BOOTSTRAP)
    stmts.extend(_NS_TABLE_DDL)
    for table in _NS_TABLES:
        for column, ctype in _NS_NODE_COLUMNS:
            stmts.append(f"ALTER TABLE {table} ADD {column} {ctype}")
    stmts.extend(_graphiti_fts_statements())
    return stmts


async def apply_ns_kuzu_schema(driver) -> None:
    """Apply the NS schema extensions; idempotent.

    ALTER TABLE ADD has no IF NOT EXISTS on the pinned Kuzu line, so re-runs
    surface already-exists binder errors — those are skipped; anything else
    propagates (a half-applied schema must fail loud, not limp).
    """
    for stmt in ns_kuzu_schema_statements():
        try:
            await driver.execute_query(stmt)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "duplicate" in msg or "already has" in msg:
                continue
            raise
    logger.info("NS Kuzu schema extensions applied")
