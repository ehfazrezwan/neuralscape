"""Profile files: ``profiles/<name>.env`` parsing and secret hygiene.

A profile is plain ``KEY=value`` lines (comments with ``#``). A value that is
a secret is never written literally (the repo is public): it is a placeholder
``<generate:KIND>`` that the gate replaces with a freshly generated value
before it starts the stack. The harness only ever reads the placeholder form,
and as a second line of defence it redacts any secret-looking key whose value
is not a placeholder before it records ``profile_flags`` in the results.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

#: ``<generate:fernet>`` (a Fernet key), ``<generate:hex32>`` (32 random bytes as hex).
PLACEHOLDER_RE = re.compile(r"^<generate:(fernet|hex32)>$")
SECRET_KEY_RE = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE)
REDACTED = "<redacted>"
HEADER_RE = re.compile(r"^#\s*(services|sidecars|mounts|requires)\s*:\s*(.*)$", re.IGNORECASE)


def _unquote(value: str) -> str:
    """Drop one pair of surrounding single or double quotes (dotenv style)."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


@dataclass
class Profile:
    flags: dict[str, str] = field(default_factory=dict)  # as recorded: placeholders kept, secrets redacted
    placeholders: dict[str, str] = field(default_factory=dict)  # KEY -> placeholder kind
    headers: dict[str, list[str]] = field(default_factory=dict)  # services / sidecars / mounts
    sha256: str | None = None

    def dictator_ids(self) -> list[str]:
        raw = self.flags.get("DICTATOR_USER_IDS", "")
        return [u.strip() for u in raw.split(",") if u.strip()]

    def first_dictator(self) -> str | None:
        ids = self.dictator_ids()
        return ids[0] if ids else None


def parse_text(text: str) -> Profile:
    prof = Profile()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = HEADER_RE.match(line)
            if m:
                items = [x.strip() for x in m.group(2).split(",") if x.strip()]
                prof.headers.setdefault(m.group(1).lower(), []).extend(items)
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), _unquote(value.strip())
        ph = PLACEHOLDER_RE.match(value)
        if ph:
            prof.placeholders[key] = ph.group(1)
        elif value and SECRET_KEY_RE.search(key):
            value = REDACTED  # never record a literal secret, whatever put it there
        prof.flags[key] = value
    return prof


def load(path: Path) -> Profile | None:
    if not path.is_file():
        return None
    raw = path.read_bytes()
    prof = parse_text(raw.decode("utf-8", "replace"))
    prof.sha256 = hashlib.sha256(raw).hexdigest()
    return prof
