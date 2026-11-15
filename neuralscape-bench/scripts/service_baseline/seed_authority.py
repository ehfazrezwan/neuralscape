"""Seed the authority store and a user token for governed-arm baseline runs.

Runs once inside the service image (``docker compose run seed``).  It binds one
benchmark user to a tenant and publishes allow statements for that user's
private and shared pools, the same shape the governed-write integration tests
use.  It also writes a signed user token for the driver.
"""

import os
import sys

sys.path.insert(0, "/app/neuralscape-service")

from tokens import issue_user_token  # noqa: E402

AUTHORITY = os.environ.get("AUTHORITY_ENABLED") == "true"
if AUTHORITY:  # images without the authority store (older trees) only need the token and the state directory
    try:
        from authority_store_sql import PostgresAuthorityStore, SqliteAuthorityStore  # noqa: E402
        from contracts_policy import PolicyStatement  # noqa: E402
        from identity_binding import bind_identity  # noqa: E402
        from policy_enforcement import pool_handle  # noqa: E402
    except ImportError as error:  # a governed arm was asked for: never record an ungoverned run under its label
        sys.exit(f"refusing to seed: the governed arm needs the authority store but this image lacks {error.name}")

USER, TENANT, PROJECT = "bench-user", "bench-tenant", "bench-project"
ACTIONS = ("read", "search", "list", "create", "update", "delete", "background_derive", "share")


def main(authority_path: str, token_path: str, secret: str) -> None:
    if not AUTHORITY:
        with open(token_path, "w") as handle:
            handle.write(issue_user_token(USER, secret, 86400))
        os.chmod("/state", 0o777)
        for name in os.listdir("/state"):
            os.chmod(os.path.join("/state", name), 0o666)
        print("seeded token only (authority store not enabled)")
        return
    if os.environ.get("AUTHORITY_BACKEND") == "postgres":
        store = PostgresAuthorityStore(os.environ["AUTHORITY_POSTGRES_DSN"])
    else:
        store = SqliteAuthorityStore(authority_path)
    subject = bind_identity(store, "local", USER).subject_id
    store.add_subject(subject, TENANT, "human", f"cred-{USER}")
    statements = []
    n = 0
    for pool_kind, owner in (("private", subject), ("shared", None)):
        for project in (None, PROJECT):
            handle = pool_handle(TENANT, pool_kind, owner, project)
            for action in ACTIONS:
                n += 1
                statements.append(
                    PolicyStatement(id=f"s{n}", effect="allow", subject_id=subject,
                                    credential_id=None, action=action, resource=handle)
                )
    store.publish_policy(TENANT, statements)
    with open(token_path, "w") as handle:
        handle.write(issue_user_token(USER, secret, 86400))
    # Services run as an unprivileged user and share this directory.
    os.chmod("/state", 0o777)
    for name in os.listdir("/state"):
        os.chmod(os.path.join("/state", name), 0o666)
    print(f"seeded subject={subject} statements={len(statements)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
