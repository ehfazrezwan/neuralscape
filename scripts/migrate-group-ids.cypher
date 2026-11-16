// Migration: legacy bare-project group_ids -> today's namespaced group_ids
//
// SCOPE: Neo4j graph only. This does not touch Qdrant. Qdrant payloads keep
// their plain `metadata.project_id` / `visibility` / `owner_user_id` fields
// (no group_id is stored there), so they need no rewrite; the graph group_id
// is rebuilt from them by memory/groups.py::_build_group_id on every write.
// Nothing else holds a legacy group_id.
//
// History. Graph rows written before the multi-user model carry a bare
// project group_id in one of two legacy shapes:
//   project:{id}   (mem0's Graphiti adapter fallback, when no explicit
//                   group_id was passed)
//   project--{id}  (the older memory service format)
// Neither is read by today's service. neuralscape-service/memory/groups.py
// (_build_group_id / _get_group_ids) builds and reads only:
//   shared--project--{id}              team pool, project scoped
//   user--{user_id}--project--{id}     private pool, project scoped
//   (each optionally suffixed --ws--{workspace}; the suffix is only added
//    for non-"memory" workspaces, so no migration is needed for it)
// so legacy rows are invisible to reads until they are rewritten.
//
// WHICH TOOL.
//   neuralscape-service/scripts/migrate_graph_groups.py --owner <uid> maps
//   project--{id} (and the bare "global") to the PRIVATE pool
//   user--<uid>--project--{id}, via the service's own driver, with a
//   dry-run default. It does NOT handle project:{id} rows.
//   This file handles BOTH legacy shapes with plain cypher-shell (both tools
//   cover node labels incl. Saga and all relationship types). The two agree when you pass the private prefix:
//     single-owner install (the normal case)  ->
//         target_prefix => 'user--<uid>--project--'   (same as --owner <uid>)
//     rows that are genuinely team knowledge  ->
//         target_prefix => 'shared--project--'
// Legacy rows carry no owner, so there is NO default: target_prefix is
// required and validated. Do not run both tools over the same rows; once a
// row is rewritten by either, the other no longer matches it.
//
// Required parameter (the script aborts, before changing anything, unless it
// is exactly 'shared--project--' or looks like 'user--<uid>--project--', with no '--' inside <uid>):
//   target_prefix
//
// NOTE: this targets the current (non-injective, `--`-joined) group-id builder in
// memory/groups.py. Ids would differ if GRAPH_GROUP_ID_INJECTIVE_ENABLED were on
// (config currently refuses it); re-check the target format before using this
// file if that flag is ever enabled.
//
// BACK UP FIRST (the rewrite is irreversible), e.g. neo4j-admin database dump
// or a volume snapshot, then take the dry run below and compare counts.
//
//   Dry run (read-only; paste into cypher-shell, no parameter needed):
//     MATCH (n) WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--' RETURN n.group_id, labels(n), count(*);
//     MATCH ()-[r]->() WHERE r.group_id STARTS WITH 'project:' OR r.group_id STARTS WITH 'project--' RETURN r.group_id, type(r), count(*);
//
//   Apply:
//     cypher-shell -u neo4j -p <password> \
//       --param "target_prefix => 'user--<uid>--project--'" \
//       < scripts/migrate-group-ids.cypher
//   Or via Docker:
//     docker compose exec -T neo4j cypher-shell -u neo4j -p <password> \
//       --param "target_prefix => 'user--<uid>--project--'" \
//       < scripts/migrate-group-ids.cypher
//
// Idempotent: rewritten ids no longer start with a legacy prefix, so a second
// run migrates nothing. Graphiti node labels: Entity, Episodic, Community,
// Saga. Relationship types carrying group_id: RELATES_TO, MENTIONS,
// HAS_MEMBER, HAS_EPISODE, NEXT_EPISODE (the relationship pass is not
// restricted to a type, so it covers all of them).

// --- Guard: abort on a bad target_prefix (1/0 raises, cypher-shell stops) ---
WITH $target_prefix AS p
WITH CASE WHEN p IS NOT NULL AND (p = 'shared--project--' OR p =~ '^user--[^-]+(-[^-]+)*--project--$') THEN 1 ELSE 0 END AS ok
RETURN 1 / ok AS target_prefix_valid;

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

// --- Migrate Saga group_ids ---
MATCH (n:Saga)
WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--'
SET n.group_id = $target_prefix + CASE WHEN n.group_id STARTS WITH 'project:' THEN substring(n.group_id, size('project:')) ELSE substring(n.group_id, size('project--')) END
RETURN 'Saga' AS label, count(*) AS migrated;

// --- Migrate relationship group_ids (RELATES_TO, MENTIONS, HAS_MEMBER, HAS_EPISODE, NEXT_EPISODE) ---
MATCH ()-[r]->()
WHERE r.group_id STARTS WITH 'project:' OR r.group_id STARTS WITH 'project--'
SET r.group_id = $target_prefix + CASE WHEN r.group_id STARTS WITH 'project:' THEN substring(r.group_id, size('project:')) ELSE substring(r.group_id, size('project--')) END
RETURN 'relationships' AS label, count(*) AS migrated;

// --- Verify: no remaining legacy-format group_ids ---
MATCH (n) WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--' RETURN count(*) AS remaining_nodes;
MATCH ()-[r]->() WHERE r.group_id STARTS WITH 'project:' OR r.group_id STARTS WITH 'project--' RETURN count(*) AS remaining_edges;
