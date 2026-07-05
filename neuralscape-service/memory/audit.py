"""Structured audit-trail logger for authoritative-context serving.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

# Structured audit trail for authoritative-context serving (standards +
# processes). Rendered as JSON in prod via logging_config; a plain stdlib
# logger is used so this has no hard dependency on structlog being configured.
import structlog  # noqa: E402

_audit_log = structlog.get_logger("neuralscape.audit")
