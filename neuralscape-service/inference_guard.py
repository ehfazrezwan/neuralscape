"""Bounded speculative compute only; callers alone publish/store results.

Timeouts cannot kill Python threads. A semaphore prevents timed-out requests
from creating an unbounded executor backlog; their eventual usage is still
logged by the provider adapter. Circuit state is process-local, never durable.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context


class InferenceGuard:
    def __init__(self, *, capacity=4, failures=3, cooldown=30.0, clock=time.monotonic):
        self._executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix='jev-extraction')
        self._slots = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self._clock, self._limit, self._cooldown = clock, failures, cooldown
        self._failures = 0
        self._open_until = 0.0
        self._probe = False
        self._epoch = 0

    def _admit(self):
        with self._lock:
            if self._failures >= self._limit:
                if self._clock() < self._open_until or self._probe:
                    return None
                self._probe = True
            return self._epoch

    def _finish(self, epoch, failed):
        with self._lock:
            if epoch != self._epoch:
                return  # an older concurrent success cannot close a new circuit
            self._probe = False
            self._failures = self._failures + 1 if failed else 0
            if self._failures >= self._limit:
                self._open_until = self._clock() + self._cooldown
                self._epoch += 1

    def run(self, operation, *, timeout, emit):
        epoch = self._admit()
        if epoch is None:
            emit({'fallback_reason': 'circuit_open'})
            return None
        if not self._slots.acquire(blocking=False):
            with self._lock:
                if epoch == self._epoch:
                    self._probe = False
            emit({'fallback_reason': 'capacity'})
            return None
        try:
            future = self._executor.submit(copy_context().run, operation)
        except Exception:
            self._slots.release()
            self._finish(epoch, True)
            raise
        future.add_done_callback(lambda _: self._slots.release())
        try:
            result, provider_failed = future.result(timeout=timeout)
        except FutureTimeout:
            self._finish(epoch, True)
            if future.done():
                raise  # operation raised TimeoutError; not an in-flight timeout
            # Provider work may still finish and bill. Never describe it as
            # cancelled or free, and never let it publish a late result.
            emit({'fallback_reason': 'deadline', 'usage_pending': True})
            return None
        except Exception:
            self._finish(epoch, True)
            raise
        self._finish(epoch, provider_failed)
        return result

    def close(self):
        self._executor.shutdown(wait=True)
