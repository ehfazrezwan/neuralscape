#!/usr/bin/env python3
"""Backfill Obsidian vault entries for memories whose ``memory_stored``
event handler failed silently due to the MemoryVisibility serialization
bug (Python 3.11+ ``str(Enum)`` regression).

Background
----------
The worker emits ``memory_stored`` events to the extension registry
after every successful memory write. The ``conversation_compiler``
extension consumes those events and appends the memory to the Obsidian
vault under ``_raw/<TypeGroup>/<Category>/...``. Before this fix, the
visibility value travelling through the event payload was the broken
string ``"MemoryVisibility.SHARED"`` (the repr of a ``(str, Enum)``
member in Python 3.11+), and the consumer's ``MemoryVisibility(...)``
constructor raised ``ValueError`` on every event — so the vault write
was silently dropped for every memory ever stored on the affected
deployment.

What this script does
---------------------
For every memory in Qdrant, construct a ``memory_stored`` event payload
that matches the shape ``worker.py`` emits (same field set, same key
names), then call ``ExtensionRegistry.emit_event("memory_stored", ...)``.
The fixed consumer's defensive parsing in
``extensions/conversation_compiler/__init__.py`` normalizes the broken
visibility values from the legacy data, so this script produces vault
entries for every memory that should have had one.

What it does NOT do
-------------------
- It does not rewrite the broken visibility values stored in Qdrant
  metadata. A separate one-shot Cypher / Qdrant scrub can do that; the
  defensive parsing means the system reads them correctly today and
  new writes are correct.
- It does not deduplicate against existing vault entries. The
  conversation_compiler's ``append_category_entry`` and
  ``append_daily_log`` are append-only. On a deployment where the bug
  caused zero vault writes, that's exactly what you want. On a
  partially-populated vault, re-running will duplicate entries.
- It does not skip ``visibility=private`` memories — the consumer
  handler filters those itself (only ``SHARED`` memories land in the
  vault). The script just emits the event for every memory and lets
  the handler decide.

Usage
-----
Run inside the running ``neuralscape`` container:

    docker exec neuralscape-neuralscape-1 \\
        python scripts/backfill_vault_writes.py [--dry-run] \\
                                                [--user USER_ID] \\
                                                [--limit N_PER_USER]

Flags:
    --dry-run         Walk everything, print stats, but emit no events.
    --user USER_ID    Limit to one user's pool (default: all users).
    --limit N         Cap memories scanned per user (default: 10000).
    --verbose         Show per-memory log lines (default: stats only).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# When invoked as ``python scripts/backfill_vault_writes.py``, Python
# puts ``scripts/`` (not the CWD) on ``sys.path[0]``, so imports from
# the service root fail. Mirror the sibling scripts (issue_user_token,
# bulk_promote_visibility, migrate_graph_groups) and prepend the
# service root explicitly so the imports below resolve.
_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

from extensions import ExtensionRegistry  # noqa: E402
from memory_service import MemoryService  # noqa: E402

logger = logging.getLogger("backfill_vault_writes")


def _payload_from_memory(mem, user_id: str) -> dict:
    """Construct a ``memory_stored`` event payload matching ``worker.py``
    emit shape for one MemoryResponse.

    ``user_id`` is the caller (matches ``worker.py``'s emit, which uses
    the request's user_id, not the memory's owner). ``owner_user_id`` is
    kept as a separate field so consumers can still distinguish them on
    legacy memories where they may differ.

    Source is set to ``"backfill"`` (NOT ``"conversation-compiler"``)
    so the consumer's skip-self-writes guard doesn't fire.
    """
    return {
        "user_id": user_id,
        "memory_id": mem.id,
        "content": mem.memory,
        "category": mem.category or "",
        "scope": mem.scope,
        "visibility": getattr(mem, "visibility", None),
        "owner_user_id": getattr(mem, "owner_user_id", None),
        "created_at": getattr(mem, "created_at", None),
        "project_id": getattr(mem, "project_id", None),
        "agent_id": None,
        "run_id": None,
        "source": "backfill",
    }


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Use the same boot path the ARQ worker uses (startup() in worker.py).
    service = MemoryService()
    registry = ExtensionRegistry()

    try:
        service._get_memory()  # warm up connections
        await registry.discover()
        await registry.startup_all()

        if args.user:
            user_ids = [args.user]
            print(f"Scoped to single user: {args.user}")
        else:
            # user_ids may contain PII (real user identifiers); log only
            # the count so backfill logs are safe to ship around.
            user_ids = service.get_all_user_ids(batch_size=100)
            print(f"Found {len(user_ids)} user(s) in pool")

        total_scanned = 0
        total_emitted = 0
        total_responses = 0
        total_errors = 0

        for uid in user_ids:
            try:
                memories = service.list_memories(user_id=uid, limit=args.limit)
            except Exception as e:
                print(f"  [user={uid}] list_memories FAILED: {e}")
                total_errors += 1
                continue

            n_user = 0
            n_resp_user = 0
            for mem in memories:
                total_scanned += 1
                n_user += 1
                payload = _payload_from_memory(mem, uid)
                if args.verbose:
                    vis = payload.get("visibility")
                    print(
                        f"  [user={uid}] {mem.id[:8]} "
                        f"[{payload.get('category','?')}/{payload.get('scope','?')}/{vis}] "
                        f"{(mem.memory or '')[:60]!r}"
                    )
                if args.dry_run:
                    continue
                try:
                    result = await registry.emit_event("memory_stored", payload)
                    total_emitted += 1
                    if result.responses:
                        n_resp_user += 1
                        total_responses += 1
                except Exception as e:
                    total_errors += 1
                    logger.warning("emit_event failed for %s: %s", mem.id, e)
            print(f"  [user={uid}] scanned={n_user} vault-writes={n_resp_user}")

        print()
        print(
            f"Summary: scanned={total_scanned} "
            f"emitted={total_emitted} "
            f"vault-writes={total_responses} "
            f"errors={total_errors}"
        )
        if args.dry_run:
            print("(dry-run — no events emitted, no vault writes)")

        return 0 if total_errors == 0 else 1
    finally:
        # Always release Redis / Qdrant / Neo4j connections, even on
        # partway failures, so a half-run leaves no dangling sockets.
        try:
            await registry.shutdown_all()
        except Exception:
            logger.exception("registry.shutdown_all failed")
        try:
            service.close()
        except Exception:
            logger.exception("service.close failed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Walk + print stats but emit no events")
    parser.add_argument("--user", type=str, default=None,
                        help="Limit to one user_id (default: all users)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Max memories scanned per user (default: 10000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-memory log lines")
    args = parser.parse_args()

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
