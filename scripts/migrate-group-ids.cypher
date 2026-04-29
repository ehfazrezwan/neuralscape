// Migration: project:{id} → project--{id} group_id format
//
// The graphiti_memory module previously used "project:{project_id}" for
// group_ids, while memory_service used "project--{project_id}". This caused
// graph search misses on project-scoped queries. After the code fix, existing
// graph data stored under the old format needs to be updated.
//
// Run against Neo4j with:
//   cypher-shell -u neo4j -p <password> < scripts/migrate-group-ids.cypher
//
// Or via Docker:
//   docker compose exec neo4j cypher-shell -u neo4j -p <password> < scripts/migrate-group-ids.cypher

// --- Dry run: preview affected nodes ---
// MATCH (n) WHERE n.group_id STARTS WITH 'project:' RETURN n.group_id, labels(n), count(*);

// --- Migrate EpisodicNode group_ids ---
MATCH (n:EpisodicNode)
WHERE n.group_id STARTS WITH 'project:'
SET n.group_id = 'project--' + substring(n.group_id, size('project:'))
RETURN 'EpisodicNode' AS label, count(*) AS migrated;

// --- Migrate EntityNode group_ids ---
MATCH (n:EntityNode)
WHERE n.group_id STARTS WITH 'project:'
SET n.group_id = 'project--' + substring(n.group_id, size('project:'))
RETURN 'EntityNode' AS label, count(*) AS migrated;

// --- Migrate edges (EntityEdge) group_ids ---
MATCH ()-[r:RELATES_TO]->()
WHERE r.group_id STARTS WITH 'project:'
SET r.group_id = 'project--' + substring(r.group_id, size('project:'))
RETURN 'RELATES_TO' AS label, count(*) AS migrated;

// --- Migrate CommunityNode group_ids ---
MATCH (n:CommunityNode)
WHERE n.group_id STARTS WITH 'project:'
SET n.group_id = 'project--' + substring(n.group_id, size('project:'))
RETURN 'CommunityNode' AS label, count(*) AS migrated;

// --- Verify: no remaining old-format group_ids ---
MATCH (n) WHERE n.group_id STARTS WITH 'project:' RETURN count(*) AS remaining_nodes;
MATCH ()-[r]->() WHERE r.group_id STARTS WITH 'project:' RETURN count(*) AS remaining_edges;
