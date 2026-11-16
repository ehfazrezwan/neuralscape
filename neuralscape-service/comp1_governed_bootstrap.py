"""Seed a disposable authority store for the governed-profile run.  Run with the TARGET tree's
interpreter and the target tree first on ``sys.path``:

  python comp1_governed_bootstrap.py --service-dir DIR --authority PATH --tenant T --users a,b \
      --projects alpha,beta

Each synthetic user is bound as a local identity and given ordinary owner grants (create, read,
list, search, update, delete, background_derive) on its own private pools: the tenant-wide pool
and one per listed project.  Nothing else is granted.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("--service-dir", "--authority", "--tenant", "--users", "--projects"):
        parser.add_argument(name, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.service_dir)
    from authority_store_sql import SqliteAuthorityStore
    from contracts_policy import PolicyStatement
    from identity_binding import bind_identity
    from policy_enforcement import pool_handle

    store = SqliteAuthorityStore(args.authority)
    actions = ("create", "read", "list", "search", "update", "delete", "background_derive")
    statements = []
    counter = 0
    for user in args.users.split(","):
        subject = bind_identity(store, "local", user).subject_id
        store.add_subject(subject, args.tenant, "human", f"cred-{user}")
        for project in [None, *args.projects.split(",")]:
            for action in actions:
                counter += 1
                statements.append(PolicyStatement(
                    id=f"comp1-{counter}", effect="allow", subject_id=subject, credential_id=None,
                    action=action, resource=pool_handle(args.tenant, "private", subject, project),
                ))
    store.publish_policy(args.tenant, statements)
    store.close()
    print("seeded", len(statements))
    return 0


if __name__ == "__main__":
    sys.exit(main())
