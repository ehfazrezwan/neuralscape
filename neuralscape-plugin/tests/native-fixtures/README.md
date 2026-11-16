# Native client fixtures

Inputs here are **synthetic and sanitized**: they follow the shapes a native client sends to
its hooks (`tool_response` objects, arrays of text blocks, nested `message.content`, tool-use and
tool-result blocks, thinking blocks, a truncated final transcript line). They contain no real
transcripts, paths, hosts or keys (paths are `/home/user/...`, URLs use `example.invalid`).
Each event fixture carries a `provenance` tag; replace a fixture with a sanitized capture when one
is available and keep its tag and expected file.

* `events/` — one PostToolUse event per file.
* `transcripts/` — session transcripts, one JSON object per line.
* `expected/` — **reviewer-owned** expected results. They are written from what a native event
  must yield (the text a later reader needs), not from what the code currently does. They are
  not edited to make a test pass; a change needs reviewer sign-off and a new `MANIFEST.json`.
* `MANIFEST.json` — SHA-256 of every file here, so an edit to a fixture or an expectation is
  visible in review.

`tests/native-fixtures.test.ts` parses each event through the hook's own code, builds a
content-free capture receipt, and compares it, and the retained text, with the expected files.
Separate boundaries (compile submission, terminal receipts, fresh-session usefulness) are not
inferred from these parser checks.
