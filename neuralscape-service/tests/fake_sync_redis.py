"""Minimal SYNC Redis stub for E3/E4 tests (lists + hashes + strings +
pipelines). Values are stored/returned as bytes, mimicking redis-py with
decode_responses=False (what session_summarizer/extraction_settings use)."""

from __future__ import annotations


def _b(value) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode()


class FakeSyncRedis:
    def __init__(self):
        self.lists: dict[str, list[bytes]] = {}
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.strings: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}

    # ── strings ──
    def get(self, key):
        return self.strings.get(key)

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.strings:
            return None
        self.strings[key] = _b(value)
        if ex:
            self.ttls[key] = ex
        return True

    def delete(self, key):
        removed = 0
        for store in (self.strings, self.lists, self.hashes):
            if key in store:
                del store[key]
                removed += 1
        return removed

    # ── lists ──
    def rpush(self, key, *values):
        lst = self.lists.setdefault(key, [])
        lst.extend(_b(v) for v in values)
        return len(lst)

    def _norm(self, n, start, stop):
        if start < 0:
            start = max(0, n + start)
        if stop < 0:
            stop = n + stop
        return start, stop

    def ltrim(self, key, start, stop):
        lst = self.lists.get(key, [])
        start, stop = self._norm(len(lst), start, stop)
        self.lists[key] = lst[start:stop + 1]
        return True

    def lrange(self, key, start, stop):
        lst = self.lists.get(key, [])
        start, stop = self._norm(len(lst), start, stop)
        return lst[start:stop + 1]

    def llen(self, key):
        return len(self.lists.get(key, []))

    # ── hashes ──
    def hincrby(self, key, field, amount=1):
        h = self.hashes.setdefault(key, {})
        f = _b(field)
        h[f] = _b(int(h.get(f, b"0")) + int(amount))
        return int(h[f])

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[_b(field)] = _b(value)
        return 1

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    # ── misc ──
    def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    def xadd(self, key, fields, maxlen=None, approximate=None):
        self.lists.setdefault(key, []).append(_b(str(fields)))
        return b"0-1"

    # ── pipeline ──
    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, parent: FakeSyncRedis):
        self.parent = parent
        self.ops: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self.ops.append((name, args, kwargs))
            return self

        return queue

    def execute(self):
        results = []
        for name, args, kwargs in self.ops:
            results.append(getattr(self.parent, name)(*args, **kwargs))
        self.ops = []
        return results
