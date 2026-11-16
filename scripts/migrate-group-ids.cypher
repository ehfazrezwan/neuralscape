// Migration: legacy bare-project group_ids -> today's namespaced group_ids
//
// History. Graph rows written before the multi-user model carry a bare
// project group_id in one of two legacy shapes:
//   project:{id}   (mem0's Graphiti adapter fallback, when no explicit
//                   group_id was passed)
//   project--{id}  (the older memory service format)
// Neither is read by today's service. neuralscape-service/memory/groups.py
// (_build_group_id / _get_group_ids) builds and reads only these ids:
//   shared--project--{id}              team pool, project scoped
//   user--{user_id}--project--{id}     private pool, project scoped
//   (each optionally suffixed --ws--{workspace}; the suffix is only added
//    for non-"memory" workspaces, so no migration is needed for it)
// so legacy rows are invisible to reads until they are rewritten.
//
// Choose the target at run time with $target_prefix. Legacy rows carry no
// owner, so the default choice is the shared pool:
//   shared pool  :  target_prefix => 'shared--project--'
//   one user's   :  target_prefix => 'user--<user_id>--project--'
//
// Run against Neo4j with:
//   cypher-shell -u neo4j -p <password> \
//     --param "target_prefix => 'shared--project--'" \
//     < scripts/migrate-group-ids.cypher
//
// Or via Docker:
//   docker compose exec -T neo4j cypher-shell -u neo4j -p <password> \
//     --param "target_prefix => 'shared--project--'" \
//     < scripts/migrate-group-ids.cypher
//
// Idempotent: rewritten ids no longer start with a legacy prefix, so a second
// run migrates nothing. Graphiti labels are Entity / Episodic / Community;
// relationships RELATES_TO / MENTIONS / HAS_MEMBER all carry group_id, so the
// relationship pass is not restricted to one type.

// --- Dry run: preview affected nodes and relationships ---
// MATCH (n) WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--' RETURN n.group_id, labels(n), count(*);
// MATCH ()-[r]->() WHERE r.group_id STARTS WITH 'project:' OR r.group_id STARTS WITH 'project--' RETURN r.group_id, type(r), count(*);

// --- Migrate Episodic group_ids ---
MATCH (n:Episodic)
WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--'
SET n.group_id = $target_prefix + CASE WHEN n.group_id STARTS WITH 'project:' THEN substring(n.group_id, size('project:')) ELSE substring(n.group_id, size('project--')) END
RETURN 'Episodic' AS label, count(*) AS migrated;

// --- Migrate Entity group_ids ---
MATCH (n:Entity)
WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--'
SET n.group_id = $target_prefix + CASE WHEN n.group_id STARTS WITH 'project:' THEN substring(n.group_id, size('project:')) ELSE substring(n.group_id, size('project--')) END
RETURN 'Entity' AS label, count(*) AS migrated;

// --- Migrate Community group_ids ---
MATCH (n:Community)
WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--'
SET n.group_id = $target_prefix + CASE WHEN n.group_id STARTS WITH 'project:' THEN substring(n.group_id, size('project:')) ELSE substring(n.group_id, size('project--')) END
RETURN 'Community' AS label, count(*) AS migrated;

// --- Migrate relationship group_ids (RELATES_TO, MENTIONS, HAS_MEMBER) ---
MATCH ()-[r]->()
WHERE r.group_id STARTS WITH 'project:' OR r.group_id STARTS WITH 'project--'
SET r.group_id = $target_prefix + CASE WHEN r.group_id STARTS WITH 'project:' THEN substring(r.group_id, size('project:')) ELSE substring(r.group_id, size('project--')) END
RETURN 'relationships' AS label, count(*) AS migrated;

// --- Verify: no remaining legacy-format group_ids ---
MATCH (n) WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--' RETURN count(*) AS remaining_nodes;
MATCH ()-[r]->() WHERE r.group_id STARTS WITH 'project:' OR r.group_id STARTS WITH 'project--' RETURN count(*) AS remaining_edges;
