"""Step ordering for the destructive backup -> wipe -> restore story, kept free of any
docker or HTTP so it is self-tested with fake steps.

Rules (a mutation that reorders or skips a step must break a self-test):
  preflight -> seed -> backup -> after_backup_write -> wipe -> gone -> restore (one retry)
  -> post checks.  Nothing is wiped unless preflight AND backup succeeded; the restore runs
  after every wipe attempt; post checks run only after a successful restore.
"""

from __future__ import annotations

from typing import Callable

import pytest

POST_STEPS = ("post_vector", "post_graph", "post_task", "post_artifact", "restarts", "fresh_write")


class Flow:
    def __init__(self) -> None:
        self.steps: dict[str, str] = {}  # name -> "" (ok) or failure text
        self.order: list[str] = []  # every step that was attempted, in order

    def run(self, name: str, fn: Callable[[], object]) -> bool:
        self.order.append(name)
        try:
            fn()
            self.steps[name] = ""
            return True
        except pytest.fail.Exception as exc:
            self.steps[name] = f"stack: {str(exc)[:300]}"
        except Exception as exc:  # noqa: BLE001  (assertion, http and docker errors are all recorded)
            self.steps[name] = f"{type(exc).__name__}: {exc}"[:400]
        return False

    def ok(self, name: str) -> bool:
        return name in self.steps and not self.steps[name]

    @property
    def wipe_attempted(self) -> bool:
        return "wipe" in self.order

    @property
    def restore_ok(self) -> bool:
        return self.ok("restore")

    @property
    def keep_backup(self) -> bool:
        """The backup is the only copy of the data once a wipe started: keep it unless restored."""
        return self.wipe_attempted and not self.restore_ok


def run_flow(flow: Flow, fns: dict[str, Callable[[], object]], scrub: Callable[[str], str] = lambda t: t) -> Flow:
    if not flow.run("preflight", fns["preflight"]):
        return flow
    seeded = flow.run("seed", fns["seed"])
    if not (seeded and flow.run("backup", fns["backup"])):
        return flow
    flow.run("after_backup_write", fns["after_backup_write"])
    if flow.run("wipe", fns["wipe"]):
        flow.run("gone", fns["gone"])
    # Restore after any wipe attempt (even a half-done one): the stack must end up on the backup.
    if not flow.run("restore", fns["restore"]):
        first = flow.steps["restore"]
        if flow.run("restore_retry", fns["restore"]):
            flow.steps["restore_first_attempt"] = first  # kept for the record; the restore itself succeeded
            flow.steps["restore"] = ""
    if flow.restore_ok:
        for name in POST_STEPS:
            flow.run(name, fns[name])
    return flow
