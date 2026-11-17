"""Every integration module refers only to names it defines or imports.

The integration modules are collected (and skipped) on a machine with no stores, so a misspelt or forgotten
import inside a test body is invisible there and only fails on the real-store run, as a ``NameError``.  No
linter runs over them, so this checks the compiled code: every global name a function loads must be bound
at module level, declared ``global``, or be a builtin.
"""

from __future__ import annotations

import builtins
import dis
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULES = sorted((ROOT / "integration_tests").glob("*.py"))


def _code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def undefined_globals(source: str, filename: str = "<module>") -> set[str]:
    """Global names loaded by some function but never bound by the module (empty for ``import *``)."""

    module = compile(source, filename, "exec")
    bound: set[str] = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__builtins__"}
    loaded: set[str] = set()
    for code in _code_objects(module):
        for ins in dis.get_instructions(code):
            if ins.opname in ("STORE_GLOBAL", "DELETE_GLOBAL"):
                bound.add(ins.argval)
            elif ins.opname == "STORE_NAME" and code is module:
                bound.add(ins.argval)
            elif ins.opname == "IMPORT_STAR":
                return set()
            elif ins.opname == "LOAD_GLOBAL" and code is not module:
                loaded.add(ins.argval)
    return loaded - bound


def test_the_check_finds_a_name_that_is_used_but_never_imported():
    source = "import os\n\ndef test_x(monkeypatch):\n    monkeypatch.setattr(engine_adapters_switch, 'A', 1)\n    os.getcwd()\n"
    assert undefined_globals(source) == {"engine_adapters_switch"}
    assert undefined_globals(source.replace("def test_x", "import engine_adapters_switch\n\ndef test_x")) == set()


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_integration_module_has_no_undefined_global_names(path):
    assert undefined_globals(path.read_text(), str(path)) == set()
