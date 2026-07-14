# Third-Party Notices — resolver-svc

This service vendors the following third-party software. Both are MIT-licensed.

## pyright

- Project: https://github.com/microsoft/pyright
- Version pinned: `1.1.411` (see `Dockerfile` `PYRIGHT_VERSION`)
- License: MIT
- Use: the `pyright-langserver` binary (installed via the `pyright` npm package)
  is driven over LSP (stdio) by this shim to resolve Python call sites to their
  definitions at index time.

## Note on multilspy

The mission brief named `multilspy` as the resolver library. We evaluated it and
found upstream `multilspy` (Microsoft) wraps **jedi-language-server** for Python —
i.e. the same Jedi-based fidelity NeuralScape already ships in-process
(`adapters/code_graph/code_resolve.py`). It has **no pyright backend**. Since the
goal is *pyright-grade* (type-checker) fidelity beyond Jedi, this service drives
`pyright-langserver` directly through a compact, purpose-built LSP client
(`lsp_client.py`) instead of adding multilspy. This is a deliberate, documented
divergence that better serves the mission goal; multilspy is therefore NOT a
dependency of this image.

- multilspy (for reference): https://github.com/microsoft/multilspy — MIT.
