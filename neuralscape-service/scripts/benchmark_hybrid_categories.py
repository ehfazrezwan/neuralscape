"""Synthetic imported-type mapping A/B using the production taxonomy and prompt."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from benchmark_hybrid import Meter, summary


async def run(args):
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from extensions.dreaming.prompts import parse_json_object
    from graphiti_core.llm_client.jev_client import JevDecisions, category_policy
    from ingest.okf_bundle import TYPE_MAPPING_PROMPT
    from schemas import CORE_MEMORY_CATEGORIES, MEMORY_CATEGORIES

    load_dotenv(args.env_file)
    expected = {'personal taste': 'preference', 'biographical detail': 'personal_fact',
                'toolchain inventory': 'tech_stack', 'coding idiom': 'convention',
                'meeting recollection': 'interaction', 'active blocker': 'task_context'}
    if args.held_out:
        expected = dict(zip([
            'preferred writing tone', 'occupation and working hours', 'programming proficiency',
            'sector expertise', 'deployed software inventory', 'repository naming rules',
            'component topology', 'third-party prerequisites', 'chosen course and rationale',
            'completed client call', 'regular release lifecycle', 'password-reset instructions',
            'unfinished investigation status'], CORE_MEMORY_CATEGORIES, strict=True))
    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
    rows = []
    arms = ['baseline', 'hybrid'] + (['scoped_gemini'] if args.scoped_control else [])
    try:
        for repeat in range(args.repeats):
            for arm in (arms if repeat % 2 == 0 else list(reversed(arms))):
                events, result = [], None
                started = time.perf_counter()
                row = {'arm': arm, 'repeat': repeat, 'events': events}
                try:
                    taxonomy = MEMORY_CATEGORIES if arm == 'baseline' else CORE_MEMORY_CATEGORIES
                    if arm == 'hybrid':
                        result = await asyncio.to_thread(JevDecisions.from_env(telemetry=events.append).classify,
                                                         list(expected), taxonomy)
                    result = result or {}
                    row['jev_accepted'] = len(result)
                    unresolved = [k for k in expected if k not in result]
                    if unresolved:
                        fallback_prompt = TYPE_MAPPING_PROMPT.format(
                            categories='\n'.join(f'- {c}' for c in sorted(taxonomy)),
                            types='\n'.join(f'- {t}' for t in unresolved))
                        if arm == 'scoped_gemini':
                            fallback_prompt += '\nUse these bucket definitions and policy:\n' + json.dumps({
                                'buckets': taxonomy, 'bucket_policy': category_policy(taxonomy)})
                        response = await Meter(client, events).generate(model='gemini-3.1-flash-lite', contents=fallback_prompt,
                            config=types.GenerateContentConfig(http_options=types.HttpOptions(timeout=60_000)))
                        result.update({k: v for k, v in parse_json_object(response.text or '').get('mapping', {}).items()
                                       if k in unresolved and v in taxonomy})
                    row.update(result=result, correct=result == expected,
                               labels_correct=sum(result.get(k) == v for k, v in expected.items()))
                except Exception as exc:
                    row.update(correct=False, labels_correct=0, error=type(exc).__name__)
                row['elapsed_ms'] = (time.perf_counter() - started) * 1000
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'events'}), flush=True)
    finally:
        await client.aio.aclose()
        client.close()
    report = {'scope': 'hand-labeled type labels; direct batched classifier, not fact extraction or alias routing',
              'held_out': args.held_out,
              'expected': expected, 'rows': rows,
              'summary': {arm: summary([r for r in rows if r['arm'] == arm]) for arm in arms}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--held-out', action='store_true')
    parser.add_argument('--scoped-control', action='store_true', help='Also compare Gemini with the same core bucket definitions and routing policy')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    asyncio.run(run(args))
