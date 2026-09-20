"""Private, synthetic component A/B. Not an end-to-end or production benchmark.

Run baseline BEFORE enabling substitutions. Outputs include every paid attempt,
including fallback usage. Labels are handwritten, never distilled from Jev.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "neuralscape-service"), str(ROOT / "graphiti"), str(ROOT / "mem0")]


def cases():
    items = []
    for name, candidate, summary, expected in [
        ("NYC", "New York City", "A city in New York", 0),
        ("Java", "Java", "An island in Indonesia", -1),
        ("Postgres", "PostgreSQL", "Relational database", 0),
        ("Alice", "Bob", "Software engineer", -1),
        ("Python", "Python", "Programming language", 0),
        ("Acme staging database", "Acme production database", "Production PostgreSQL", -1),
    ]:
        items.append({"id": f"entity-{len(items)}", "task": "nodes", "context": {
            "extracted_nodes": [{"id": 0, "name": name, "entity_type": "Entity"}],
            "existing_nodes": [{"candidate_id": 0, "name": candidate, "summary": summary, "entity_types": ["Entity"]}],
            "episode_content": f"We use {name} as a programming language." if name == "Java" else f"We discussed {name}.",
            "previous_episodes": []}, "expected": [expected]})
    for old, new, duplicate, contradiction in [
        ("Alice works at Acme.", "Alice is employed by Acme.", [0], []),
        ("Bob ran 5 miles on Tuesday.", "Bob ran 3 miles on Wednesday.", [], []),
        ("The API timeout is 30 seconds.", "The API timeout is now 60 seconds, replacing 30 seconds.", [], [0]),
        ("Alice prefers Python.", "Bob prefers Python.", [], []),
        ("Release v1 used SQLite.", "Release v2 uses PostgreSQL.", [], []),
        ("The cache TTL is 60 seconds.", "The cache TTL is 60 seconds.", [0], []),
    ]:
        items.append({"id": f"edge-{len(items)}", "task": "edges", "context": {
            "existing_edges": [{"idx": 0, "fact": old}], "edge_invalidation_candidates": [], "new_edge": new},
            "expected": {"duplicate_facts": duplicate, "contradicted_facts": contradiction}})
    return items


def challenge_cases():
    """Additional frozen hand labels: homonyms, qualifiers, negation, versions."""
    items = []
    for name, episode, candidates, expected in [
        ('Paris', 'We are visiting Paris in Texas, USA.', [('Paris', 'Capital of France'), ('Paris', 'City in Texas, USA')], 1),
        ('Jordan', 'Jordan is the designer on the Maple team.', [('Jordan', 'Engineer on Cedar team'), ('Jordan', 'Designer on Maple team')], 1),
        ('ML', 'ML is short for machine learning in this episode.', [('MetroLink', 'Rail transit operator'), ('Machine Learning', 'Statistical learning field')], 1),
        ('Acme test cluster', 'The test cluster is separate from the live customer cluster.', [('Acme production cluster', 'Hosts live customers')], -1),
    ]:
        items.append({'id': f'challenge-node-{len(items)}', 'task': 'nodes', 'context': {
            'extracted_nodes': [{'id': 0, 'name': name, 'entity_type': 'Entity'}],
            'existing_nodes': [{'candidate_id': i, 'name': n, 'summary': s, 'entity_types': ['Entity']}
                               for i, (n, s) in enumerate(candidates)],
            'episode_content': episode, 'previous_episodes': []}, 'expected': [expected]})
    for old, new, duplicate, contradiction in [
        ('Ada permits analytics tracking.', 'Ada no longer permits analytics tracking.', [], [0]),
        ('The production timeout is 30 seconds.', 'The staging timeout is 60 seconds.', [], []),
        ('The maximum request size is 10 MB.', 'The maximum request size is now 20 MB instead of 10 MB.', [], [0]),
        ('Version 1 encrypts with AES-128.', 'Version 2 encrypts with AES-256.', [], []),
        ('Ada owns the deployment keys.', 'Bob owns his deployment keys.', [], []),
        ('The retry limit is three attempts.', 'The retry limit is 3 attempts.', [0], []),
        ('Ada prefers email, not calls.', 'Ada prefers email.', [], []),
        ('The service is not available in Canada.', 'The service is now available in Canada.', [], [0]),
    ]:
        items.append({'id': f'challenge-edge-{len(items)}', 'task': 'edges', 'context': {
            'existing_edges': [{'idx': 0, 'fact': old}], 'edge_invalidation_candidates': [], 'new_edge': new},
            'expected': {'duplicate_facts': duplicate, 'contradicted_facts': contradiction}})
    return items


class Meter:
    """Wrap SDK I/O, not model-internal trackers (which can miss thought tokens)."""
    def __init__(self, client, events):
        self.client, self.events = client, events
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate))
        self.models = SimpleNamespace(generate_content=self.generate_sync)

    @contextmanager
    def measured(self, model):
        started = time.perf_counter()
        event = {"provider": "gemini", "model": model, "input_tokens": None, "output_tokens": None}
        try:
            yield event
        except Exception as exc:
            event["error"] = type(exc).__name__  # never serialize request/key/body
            raise
        finally:
            event["elapsed_ms"] = (time.perf_counter() - started) * 1000
            self.events.append(event)

    @staticmethod
    def record_usage(response, event):
        usage = response.usage_metadata
        if usage:
            event.update(input_tokens=usage.prompt_token_count or 0,
                         output_tokens=(usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
                         cached_tokens=usage.cached_content_token_count or 0)

    async def generate(self, **kwargs):
        with self.measured(kwargs['model']) as event:
            response = await self.client.aio.models.generate_content(**kwargs)
            self.record_usage(response, event)
            return response

    def generate_sync(self, **kwargs):
        with self.measured(kwargs['model']) as event:
            response = self.client.models.generate_content(**kwargs)
            self.record_usage(response, event)
            return response


async def execute(case, args, client):
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.gemini_client import GeminiClient
    from graphiti_core.prompts import dedupe_edges, dedupe_nodes

    events = []
    metered = Meter(client, events)
    fallback = GeminiClient(LLMConfig(model=args.model, small_model=args.model), client=metered)
    started = time.perf_counter()
    row = {"id": case["id"], "task": case["task"], "events": events}
    try:
        module = dedupe_nodes if case["task"] == "nodes" else dedupe_edges
        prompt = module.nodes if case["task"] == "nodes" else module.resolve_edge
        schema = module.NodeResolutions if case["task"] == "nodes" else module.EdgeDuplicate
        result = None
        if args.arm == "hybrid":
            from graphiti_core.llm_client.jev_client import JevDecisions
            decisions = JevDecisions.from_env(telemetry=events.append)
            result = await decisions.resolve(case["task"], case["context"])
        if result is None:
            result = await fallback.generate_response(prompt(case["context"]), response_model=schema)
        actual = [r["duplicate_candidate_id"] for r in result["entity_resolutions"]] if case["task"] == "nodes" else result
        row["correct"] = actual == case["expected"]
        row["result"] = result  # only the public synthetic fixtures, never live memories
    except Exception as exc:
        row.update(correct=False, error=type(exc).__name__)
    row["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return row


def summary(rows):
    durations = sorted(r["elapsed_ms"] for r in rows)
    events = [e for r in rows for e in r["events"] if "input_tokens" in e and
              (e.get('provider') != 'jev' or e.get('attempted'))]
    known = [e for e in events if e["input_tokens"] is not None and e.get("output_tokens") is not None]
    rates = {"gemini-3.1-flash-lite": (0.25, 1.5), "jev-1.13.0": (0.042, 0)}
    unknown = [e for e in events if e not in known or e["model"] not in rates]
    # A timed-out thread can still finish its HTTP request after this snapshot.
    # Never report its possible paid usage as a fully measured zero.
    pending = sum(e.get('fallback_reason') == 'deadline' for r in rows for e in r['events'])
    cost = sum((e["input_tokens"] * rates[e["model"]][0] + e["output_tokens"] * rates[e["model"]][1]) / 1e6
               for e in known if e["model"] in rates)
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "fallback_events": sum('fallback_reason' in e for r in rows for e in r['events']),
            "errors": sum("error" in r for r in rows),
            "p50_ms": statistics.median(durations), "p95_ms": durations[math.ceil(.95 * len(durations)) - 1],
            "known_list_price_usd": cost, "cost_complete": not unknown and not pending,
            "deadline_usage_pending": pending,
            "unpriced_or_unknown_usage_calls": len(unknown), "provider_calls": len(events),
            "input_tokens": sum(e["input_tokens"] for e in known),
            "output_tokens": sum(e["output_tokens"] for e in known)}


async def run(args):
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(args.env_file, override=False)
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is required")
    fixtures = challenge_cases() if args.challenge else cases()
    if args.limit:
        fixtures = fixtures[:args.limit]
    rows = []
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    try:
        # Sequential arms avoid hiding latency behind different concurrency.
        for repeat in range(args.repeats):
            for case in fixtures:
                row = await execute(case, args, client)
                row["repeat"] = repeat
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ("result", "events")}), flush=True)
                report = {"arm": args.arm, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "fixture_sha256": hashlib.sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest(),
                    "platform": platform.platform(), "scope": "synthetic component pilot; excludes stores, queues, embeddings, answers",
                    "pricing": "Standard USD list prices 2026-09-21; not invoice cost; no cache discount assumed",
                    "summary": summary(rows), "by_task": {t: summary([r for r in rows if r["task"] == t]) for t in {r["task"] for r in rows}}, "rows": rows}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        await client.aio.aclose()
        client.close()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["baseline", "hybrid"], default="baseline")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--challenge", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.limit < 0:
        parser.error("repeats must be positive and limit nonnegative")
    asyncio.run(run(args))
