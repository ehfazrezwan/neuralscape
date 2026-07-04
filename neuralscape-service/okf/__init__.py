"""Open Knowledge Format (OKF) edge interop.

Boundary features only — OKF defines no taxonomy or ontology, so this is
an envelope concern, deliberately NOT a knowledge adapter:

- :mod:`okf.translate` — THE translation module. Every OKF key name,
  reserved filename, version string, and the category↔type mapping lives
  here and nowhere else.
- :mod:`okf.conformance` — §9 conformance walker (used by tests + E2E).
- :mod:`okf.vault` — renders the dreaming vault's bundle metadata
  (per-folder index files, root version marker, sweep history log).
- :mod:`okf.export` — builds a spec-conformant bundle zip from the
  memory store for ``GET /v1/export/okf``.
"""
