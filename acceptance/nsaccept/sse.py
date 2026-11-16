"""Server-sent events reader for ``GET /v1/stream``.

``SSEReader`` opens the stream on a background thread and keeps every parsed
frame, so a scenario can (1) open the stream, (2) wait until the server has
subscribed (the first ``: connected`` comment), (3) cause an event, and
(4) wait for it, without blocking on the endless response body.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import httpx


@dataclass
class SSEFrame:
    event: str = "message"
    data: str = ""
    comment: str | None = None  # set for ``: ...`` comment frames (connected / keep-alive)

    def json(self) -> dict | None:
        try:
            parsed = json.loads(self.data)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SSEFrame]:
    """Parse an SSE text stream (already split into lines) into frames."""
    event, data, comment = "message", [], None
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data:
                yield SSEFrame(event=event, data="\n".join(data))
            elif comment is not None:
                yield SSEFrame(comment=comment)
            event, data, comment = "message", [], None
        elif line.startswith(":"):
            comment = line[1:].strip()
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    # a trailing frame without the blank-line terminator is dropped on purpose


@dataclass
class SSEReader:
    """Background reader of one SSE endpoint."""

    base_url: str
    token: str | None
    path: str = "/v1/stream"
    frames: list[SSEFrame] = field(default_factory=list)
    status_code: int | None = None
    error: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> "SSEReader":
        self._thread = threading.Thread(target=self._run, daemon=True, name="ns-sse-reader")
        self._thread.start()
        return self

    def _run(self) -> None:
        headers = {"Accept": "text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        timeout = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)
        try:
            with httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout) as client:
                with client.stream("GET", self.path) as resp:
                    self.status_code = resp.status_code
                    if resp.status_code != 200:
                        self.error = f"HTTP {resp.status_code}"
                        return
                    for frame in parse_sse_lines(resp.iter_lines()):
                        self.frames.append(frame)
                        if self._stop.is_set():
                            return
        except Exception as exc:  # noqa: BLE001 - recorded, surfaced by wait_for / close
            if not self._stop.is_set():
                self.error = f"{type(exc).__name__}: {exc}"

    def wait_for(
        self,
        predicate: Callable[[SSEFrame], bool],
        timeout: float = 60.0,
        interval: float = 0.25,
    ) -> SSEFrame | None:
        """First frame matching ``predicate`` within ``timeout`` seconds, else None."""
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            while seen < len(self.frames):
                frame = self.frames[seen]
                seen += 1
                if predicate(frame):
                    return frame
            if self.error or time.monotonic() >= deadline:
                return None
            time.sleep(interval)

    def wait_connected(self, timeout: float = 15.0) -> bool:
        """True once the server sent its ``: connected`` frame (it has subscribed)."""
        return self.wait_for(lambda f: f.comment == "connected", timeout) is not None

    def events(self) -> list[dict]:
        return [j for f in list(self.frames) if (j := f.json()) is not None]

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
