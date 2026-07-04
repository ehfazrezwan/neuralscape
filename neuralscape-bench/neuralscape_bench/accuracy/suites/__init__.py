"""Suite registry: name → fetch/load callables + metadata.

A ``Suite`` binds a dataset fetcher (network) to a pure parser (unit-tested
on fixtures). ``load`` receives the on-disk dataset dir plus sampling
options and returns a normalized :class:`~..schema.SuiteData`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from neuralscape_bench.accuracy.schema import SuiteData


class LoadFn(Protocol):
    def __call__(self, dest_dir: Path, *, sample: int | None = None,
                 seed: int = 42, options: dict | None = None) -> SuiteData: ...


@dataclass(frozen=True)
class Suite:
    name: str                       # CLI id, e.g. "longmemeval_s"
    display: str                    # human label for reports
    fetch: Callable[[Path], dict]   # download into dest dir → verify info
    load: "LoadFn"
    source: str                     # canonical dataset origin (URL / repo)
    license_note: str = ""
    default_options: dict = field(default_factory=dict)


def get_suite(name: str) -> Suite:
    from neuralscape_bench.accuracy.suites import (  # noqa: PLC0415 — lazy, avoids import cycles
        beam, convomem, dmr, locomo, longmemeval, membench,
    )

    registry: dict[str, Suite] = {}
    for mod in (locomo, longmemeval, dmr, beam, convomem, membench):
        for s in mod.SUITES:
            registry[s.name] = s
    if name not in registry:
        raise KeyError(f"Unknown suite {name!r}. Known: {sorted(registry)}")
    return registry[name]


def all_suite_names() -> list[str]:
    """Every registered suite id, in report order."""
    return [
        "locomo",
        "longmemeval_s",
        "longmemeval_m",
        "dmr",
        "beam",
        "convomem",
        "membench",
    ]
