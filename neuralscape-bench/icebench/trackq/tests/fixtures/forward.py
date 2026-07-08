"""Fixture exercising forward references (callee defined AFTER caller)."""


def early_caller():
    """Calls a function defined later in the file."""
    return late_callee()


def late_callee():
    """Defined after early_caller — a forward reference target."""
    return 42
