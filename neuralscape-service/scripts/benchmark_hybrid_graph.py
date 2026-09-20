"""Real Neo4j + Qdrant service-path pilot on isolated loopback stores ONLY.

No destructive cleanup. Each arm uses unique user/collection namespaces.
Graph generation usage is measured; embedding costs remain explicitly unpriced.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from benchmark_hybrid import Meter, summary


def run(args):
    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    from config import settings
    from memory_service import MemoryService
    import hybrid_inference

    run_id = 'hybrideval' + uuid.uuid4().hex[:10]
    # Hard-coded test endpoints avoid accidental writes into an operator's DB.
    settings.neo4j_uri = 'bolt://127.0.0.1:17687'
    settings.neo4j_user, settings.neo4j_password, settings.neo4j_database = 'neo4j', 'test-only', 'neo4j'
    settings.qdrant_url = 'http://127.0.0.1:16333'
    settings.qdrant_collection = run_id
    settings.redis_url = 'redis://127.0.0.1:16379/0'
    settings.llm_gateway_enabled = False
    settings.llm_gateway_graphiti_enabled = False
    settings.jev_graph_enabled = args.arm == 'hybrid'
    settings.jev_rerank_enabled = False
    settings.jev_extraction_enabled = args.extract and args.arm == 'hybrid'
    settings.typesafe_api_key = os.environ.get('TYPESAFE_API_KEY', '')
    settings.google_api_key = os.environ['GOOGLE_API_KEY']
    if args.simulate_jev_outage:
        settings.typesafe_api_key = ''
    settings.reinforcement_boost_k = 0
    service = MemoryService()
    rows, checks, events = [], {}, []
    factory = hybrid_inference.decisions
    hybrid_inference.decisions = lambda **kw: factory(telemetry=events.append)
    try:
        if args.extract:
            service._genai_model = Meter(service._get_genai_client(), events)
        memory = service._get_memory()
        hybrid_inference.warmup_extraction()
        assert service._graphiti is not None, 'Graph must initialize; vector-only is a failure'
        graph = service._graphiti
        metered = Meter(graph.llm_client.client, events)
        graph.llm_client.client = metered
        decisions = getattr(graph.llm_client, 'decision_client', None)
        if decisions:
            decisions.telemetry = events.append
            if args.simulate_jev_outage:
                decisions._api_key = ''
        facts = [
            ('Ada works at Acme as of January 1, 2025.', '2025-01-01T00:00:00+00:00'),
            ('Ada is employed by Acme as of January 1, 2025.', '2025-01-01T00:00:00+00:00'),
            ('On June 1, 2025, Ada left Acme and now works at Beta.', '2025-06-01T00:00:00+00:00'),
        ]
        ids = []
        extracted = []
        evidence_ids = []
        for index, (fact, date) in enumerate(facts):
            start, offset = time.perf_counter(), len(events)
            evidence = {}
            selected = service.extract_facts_only(fact, category_evidence=evidence) if args.extract else [('personal_fact', fact)]
            extracted.append(selected)
            result = []
            for category, content in selected:
                stored = service.store_raw(content=content, category=category, user_id=run_id,
                              visibility='private', occurred_at=date, add_to_graph=True,
                              category_evidence=evidence.get((category, content)))
                result.extend(stored)
                if (category, content) in evidence:
                    evidence_ids.extend((r.id, evidence[(category, content)]) for r in stored)
            ids.extend(r.id for r in result)
            row = {'id': f'write-{index}', 'task': 'store_raw_plus_graph', 'correct': bool(result),
                   'elapsed_ms': (time.perf_counter() - start) * 1000, 'events': events[offset:]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'events'}), flush=True)
        # Exact idempotence must avoid another graph episode/model call.
        offset = len(events)
        again = service.store_raw(content=extracted[0][0][1], category=extracted[0][0][0], user_id=run_id,
                                  visibility='private', occurred_at=facts[0][1], add_to_graph=True)
        checks['idempotent_write'] = bool(again) and again[0].id == ids[0] and len(events) == offset
        if evidence_ids:
            checks['category_evidence_roundtrip'] = all(
                service._point_to_response(memory.vector_store.client.retrieve(
                    collection_name=run_id, ids=[mid], with_payload=True)[0]).category_evidence == expected
                for mid, expected in evidence_ids)
        for vector_only in (True, False):
            found = service.search(query='Where does Ada work?', user_id=run_id, limit=10,
                                   vector_only=vector_only, include_shared=False)
            checks['vector_recall' if vector_only else 'combined_recall'] = any('Beta' in r.memory for r in found)
            foreign = service.search(query='Where does Ada work?', user_id=run_id + 'other', limit=10,
                                     vector_only=vector_only, include_shared=False)
            checks['vector_isolation' if vector_only else 'combined_isolation'] = not foreign
        # Query actual persisted graph, not just add()'s return or vector recall.
        async def inspect():
            records, _, _ = await graph.driver.execute_query(
                'MATCH (n:Entity) WHERE n.group_id CONTAINS $run RETURN n.name AS name, n.group_id AS group_id',
                run=run_id)
            edges, _, _ = await graph.driver.execute_query(
                'MATCH ()-[e:RELATES_TO]->() WHERE e.group_id CONTAINS $run '
                'RETURN e.fact AS fact, e.valid_at AS valid_at, e.invalid_at AS invalid_at', run=run_id)
            return [dict(r) for r in records], [dict(r) for r in edges]
        nodes, edges = service._bridge.run(inspect())
        checks['graph_persisted'] = bool(nodes) and bool(edges)
        checks['entity_reuse'] = sum(n['name'].casefold() == 'ada' for n in nodes) == 1
        checks['old_relationship_invalidated'] = any('Acme' in e['fact'] and e['invalid_at'] is not None for e in edges)
        checks['new_relationship_current'] = any('Beta' in e['fact'] and e['invalid_at'] is None for e in edges)
        from memory.groups import _build_group_id
        own_group = _build_group_id('private', run_id)
        checks['graph_group_isolation'] = all(n['group_id'] == own_group for n in nodes)
        groups = list({n['group_id'] for n in nodes})
        graph_hits = memory.graph.search('Where does Ada work?', {'group_ids': groups}, limit=10)
        checks['graph_only_recall'] = any('Beta' in r.get('fact', '') for r in graph_hits)
        checks['graph_only_isolation'] = not memory.graph.search('Where does Ada work?', {'group_ids': [run_id + 'other']}, limit=10)
        # Populate competing namespaces with the SAME entity name. An empty
        # foreign namespace alone cannot reveal cross-tenant merge/invalidation.
        validation_offset = len(events)
        for user, project, employer in [(run_id + 'other', None, 'Orion'), (run_id, 'other-project', 'Vega')]:
            service.store_raw(content=f'Ada works at {employer}.', category='personal_fact',
                              user_id=user, scope='project' if project else 'global', project_id=project,
                              visibility='private', add_to_graph=True)
            foreign_group = _build_group_id('private', user, project)
            foreign_hits = memory.graph.search('Where does Ada work?', {'group_ids': [foreign_group]}, limit=10)
            label = 'project' if project else 'tenant'
            checks[f'populated_{label}_graph_recall'] = any(employer in r.get('fact', '') for r in foreign_hits)
            checks[f'populated_{label}_graph_isolation'] = not any('Beta' in r.get('fact', '') for r in foreign_hits)
        own_hits = memory.graph.search('Where does Ada work?', {'group_ids': [own_group]}, limit=10)
        checks['own_graph_excludes_foreign_entities'] = not any(
            any(name in r.get('fact', '') for name in ('Orion', 'Vega')) for r in own_hits)
        for vector_only in (True, False):
            own_hits = service.search(query='Where does Ada work?', user_id=run_id,
                                      project_id='target-project', vector_only=vector_only,
                                      include_shared=False, limit=20)
            checks['populated_vector_isolation' if vector_only else 'populated_combined_isolation'] = (
                any('Beta' in r.memory for r in own_hits) and
                not any(any(name in r.memory for name in ('Orion', 'Vega')) for r in own_hits))
        async def isolated_edges():
            records, _, _ = await graph.driver.execute_query(
                'MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity) WHERE e.group_id = $group '
                'RETURN a.group_id AS source_group, b.group_id AS target_group, '
                'e.fact AS fact, e.invalid_at AS invalid_at', group=own_group)
            return [dict(r) for r in records]
        after = service._bridge.run(isolated_edges())
        checks['no_cross_group_endpoints'] = bool(after) and all(
            r['source_group'] == own_group == r['target_group'] for r in after)
        checks['foreign_write_cannot_invalidate_own_graph'] = any(
            'Beta' in r['fact'] and r['invalid_at'] is None for r in after)
        # Deterministic soft-score fixture checks actual storage semantics even
        # if this provider run happens to return only confident classifications.
        from category_evidence import make_evidence
        from schemas import CORE_MEMORY_CATEGORIES
        text = 'Project Cedar uses Redis for caching, not durable storage.'
        probabilities = dict.fromkeys([*CORE_MEMORY_CATEGORIES, 'UNCERTAIN'], 0.0)
        probabilities.update(tech_stack=.6, architecture=.4)
        soft = make_evidence(text, {'choice':'tech_stack', 'confidence':.55,
                                   'probabilities':probabilities}, 'jev-1.13.0', 'soft')
        soft_user = run_id + 'soft'
        result = service.store_raw(text, soft_user, 'tech_stack', scope='project', project_id='cedar',
                                   visibility='private', category_evidence=soft, add_to_graph=True)
        point = memory.vector_store.client.retrieve(collection_name=run_id, ids=[result[0].id], with_payload=True)[0]
        checks['soft_evidence_persisted'] = service._point_to_response(point).category_evidence == soft
        checks['soft_evidence_stays_private'] = point.payload['metadata']['visibility'] == 'private'
        checks['soft_secondary_does_not_expand_strict_filter'] = not service.search(
            query='Redis caching', user_id=soft_user, project_id='cedar', categories=['architecture'],
            vector_only=True, include_shared=False)
        soft_group = _build_group_id('private', soft_user, 'cedar')
        soft_hits = memory.graph.search('Redis caching', {'group_ids':[soft_group]}, limit=10)
        checks['soft_fact_graph_recall'] = any('Redis' in hit.get('fact', '') for hit in soft_hits)
        checks['soft_fact_foreign_graph_isolation'] = not memory.graph.search(
            'Redis caching', {'group_ids':[_build_group_id('private', soft_user + 'other', 'cedar')]}, limit=10)
        report = {'arm': args.arm, 'run_id': run_id, 'checks': checks,
                  'extraction_enabled': args.extract, 'extracted': extracted,
                  'extracted_evidence_count': len(evidence_ids),
                  'summary': summary(rows), 'rows': rows, 'nodes': nodes, 'edges': edges,
                  'extra_validation_events': events[validation_offset:],
                  'scope': 'real MemoryService writes with optional preceding extraction, Mem0 adapter, Neo4j and Qdrant; excludes API/queue time; generation cost only, embeddings unpriced'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, default=str, indent=2) + '\n')
        print(json.dumps(checks, indent=2))
        if not all(checks.values()):
            raise SystemExit('Graph/vector acceptance checks failed; inspect the report')
    finally:
        hybrid_inference.decisions = factory
        service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['baseline', 'hybrid'], required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--simulate-jev-outage', action='store_true')
    parser.add_argument('--extract', action='store_true', help='Extract facts before persisting to both stores')
    run(parser.parse_args())
