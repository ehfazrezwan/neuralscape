"""API launcher for the baseline stack.

The governed tree installs ``policy_enforcement`` hooks on every route but has
no production code that configures the enforcer, so ``NS_POLICY_MODE=enforce``
without wiring denies every operation as ``authority_unavailable``.  When
``HARNESS_WIRE_ENFORCER=1`` this launcher wires the enforcer the way the
integration tests do (authority store as both authority and binding source)
before the app starts, so the cost of evaluating the real decision path can be
measured.  Without it the API starts exactly as in production.
"""

import os
import sys

sys.path.insert(0, "/app/neuralscape-service")
os.chdir("/app/neuralscape-service")

if os.environ.get("HARNESS_WIRE_ENFORCER") == "1":
    import policy_enforcement
    from authority_store_sql import SqliteAuthorityStore
    from policy_enforcement import PolicyEnforcer

    store = SqliteAuthorityStore(os.environ["AUTHORITY_SQLITE_PATH"])
    policy_enforcement.configure(PolicyEnforcer(store, store))

import uvicorn  # noqa: E402


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8199, workers=int(os.environ.get("UVICORN_WORKERS", "1")))


if __name__ == "__main__":  # worker processes re-import this module and must not start another server
    main()
