# `neuralscape-service/vendor/`

This directory is intentionally empty apart from `.gitkeep` and this file. It
exists so the Dockerfile's `COPY neuralscape-service/vendor/ neuralscape-service/vendor/`
(in the builder and test stages) always has a source to copy; Docker fails the
build if a COPY source is missing.

What it is for: an optional, pre-downloaded `en_core_web_sm-*.whl` (the spaCy
model that mem0's keyword layer loads). When exactly one such wheel is present
the build installs it instead of running `python -m spacy download`, which
matters on hosts whose TLS is intercepted and cannot reach the model's release
host. With no wheel, the build downloads the model normally. Two or more wheels
fail the build on purpose. `*.whl` files here are git-ignored, so a wheel is a
local build input and is never committed.

Do not delete the directory or `.gitkeep`. This README does not affect the
build: the Dockerfile only globs `vendor/en_core_web_sm-*.whl`.
