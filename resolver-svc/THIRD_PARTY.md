# Third-Party Notices — resolver-svc

This service does not vendor third-party source: pyright is installed from npm and the Python packages from PyPI when the image is built. The following third-party software ends up in the image. Each keeps its own license.

## pyright

- Project: https://github.com/microsoft/pyright
- Version pinned: `1.1.411` (see `Dockerfile` `PYRIGHT_VERSION`)
- License: MIT
- Use: the `pyright-langserver` binary (installed via the `pyright` npm package)
  is driven over LSP (stdio) by this shim to resolve Python call sites to their
  definitions at index time.

## Other components in the image

| Component | How it arrives | License |
|---|---|---|
| Node.js (`node:20-slim` base image) | image base layer | MIT, with bundled components under their own licenses (see the Node.js project) |
| FastAPI, pydantic, anyio | `pip install` in the Dockerfile (versions pinned there) | MIT |
| uvicorn | `pip install` in the Dockerfile (version pinned there) | BSD-3-Clause |
| Debian packages (`python3`, `curl`, and their dependencies) | `apt-get install` in the Dockerfile | their own Debian-packaged licenses |

The main service's `adapters/code_graph/code_resolve.py` uses Jedi (MIT,
a normal Python dependency) in process; Jedi is not part of this image.
The repository-level notice file is `THIRD_PARTY_NOTICES.md`.

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
