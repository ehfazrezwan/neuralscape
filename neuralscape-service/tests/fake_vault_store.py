"""In-memory async KV with the Redis commands ``ConnectorVault`` uses.

``eval`` serves the vault's two compare-and-set scripts (recognised by the
marker comment on their first line) the way Redis runs them: the check and
the write happen with no await in between, so nothing can interleave.

``pause_after_get`` lets a test stop a vault call right after it has read a
key, run other writes, then let it continue, to interleave deterministically
between a read and the write computed from it.
"""

from __future__ import annotations

import asyncio
import fnmatch


class FakeVaultStore:
    def __init__(self):
        self.d: dict[str, str] = {}
        self._pause_key: str | None = None
        self.paused = asyncio.Event()
        self.resume = asyncio.Event()

    def pause_after_get(self, key: str) -> None:
        """Pause the next ``get`` of ``key`` after it has read the value."""
        self._pause_key = key

    async def get(self, k):
        value = self.d.get(k)
        if self._pause_key == k:
            self._pause_key = None
            self.paused.set()
            await self.resume.wait()
        return value

    async def set(self, k, v):
        self.d[k] = v

    async def delete(self, k):
        return 1 if self.d.pop(k, None) is not None else 0

    async def keys(self, pattern):
        return [k for k in self.d if fnmatch.fnmatch(k, pattern)]

    async def eval(self, script, numkeys, *args):
        keys, argv = list(args[:numkeys]), list(args[numkeys:])
        marker = script.lstrip().splitlines()[0]
        cur = self.d.get(keys[0])
        if marker == "-- ns-vault:cas-set":
            must_exist, expected, value = argv
            if must_exist == "1":
                if cur != expected:
                    return 0
            elif cur is not None:
                return 0
            self.d[keys[0]] = value
            return 1
        if marker == "-- ns-vault:cas-delete":
            if cur is not None and cur == argv[0]:
                del self.d[keys[0]]
                return 1
            return 0
        raise NotImplementedError(f"unexpected script: {marker!r}")
