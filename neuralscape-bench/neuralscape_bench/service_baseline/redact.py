"""Redact credentials from run records by value, not only by key name."""

from __future__ import annotations

import re
from typing import Any

_SENSITIVE_KEY = re.compile(r"(password|passwd|secret|token|api[_-]?key)", re.IGNORECASE)
# key=value and key='quoted value' pairs in connection strings, and SQL "PASSWORD 'value'".
_PAIR = re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)(\s*=\s*)('[^']*'|\"[^\"]*\"|\S+)")
_SQL_PASSWORD = re.compile(r"(?i)\bPASSWORD\s+'[^']*'")
REDACTED = "<redacted>"


def redact_text(text: str) -> str:
    text = _SQL_PASSWORD.sub(f"PASSWORD '{REDACTED}'", text)
    return _PAIR.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


def redact_value(value: Any) -> Any:
    """Return a copy with credential values removed, whatever the key is called."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            key: REDACTED if isinstance(key, str) and _SENSITIVE_KEY.search(key) and isinstance(item, str)
            else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    return value
