#!/usr/bin/env python3
"""RETIRED: the Wiki/ by-topic -> by-scope layout migration.

This script pivoted ``Wiki/<TypeGroup>/<CategoryFolder>/<file>.md`` pages to
``Wiki/<scope>/<TypeGroup>/<CategoryLeaf>.md``. It depended on the path
builder in ``extensions.wiki_synthesizer.wiki_renderer``.

Commit 780b933a ("feat(api)!: retire wiki_synthesizer -- dreaming supersedes
it") deleted that extension (renderer, synthesizer, tests and the
/v1/admin/synthesize endpoints). Nothing writes either Wiki/ layout any more:
the dreaming librarian writes ``Projects/<project>/<Topic>.md``,
``Knowledge/<Topic>.md``, ``Me/<Topic>.md`` and ``Home.md``. The script's
import target no longer exists, so it cannot run, and the old-layout pages it
acted on can no longer be produced.

For a vault that still holds a legacy ``Wiki/`` tree, use the supported
successor ``neuralscape-service/scripts/migrate_wiki_to_topics.py``, which
converts ``Wiki/<scope>/<TypeGroup>/<CategoryLeaf>.md`` into the librarian
layout and archives (never deletes) the legacy pages.

Running this script exits with status 2, prints the message below to stderr
and touches nothing.
"""

from __future__ import annotations

import sys

RETIRING_COMMIT = "780b933a"
SUCCESSOR = "neuralscape-service/scripts/migrate_wiki_to_topics.py"

MESSAGE = (
    "migrate_wiki_layout.py is retired: the wiki_synthesizer extension it "
    f"depended on was removed in commit {RETIRING_COMMIT} (dreaming supersedes "
    "it), so nothing writes the old or new Wiki/ layouts any more. No vault "
    "files were touched. To convert a legacy Wiki/ tree to the librarian "
    f"layout use {SUCCESSOR}."
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
