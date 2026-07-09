"""
Tests for corpus filtering and preprocessing utilities.
"""

import pytest
from pathlib import Path
from icebench.corpus_filters import (
    is_tool_output,
    strip_python_docstrings,
    TOOL_OUTPUT_DIRS,
)


def test_is_tool_output():
    """Test tool output directory detection."""
    root = Path("/data/ice/corpora/small-py@abc123")

    # Tool output files should be detected
    assert is_tool_output(root / "graphify-out" / "graph.json", root)
    assert is_tool_output(root / "src" / "graphify-out" / "foo.py", root)
    assert is_tool_output(root / "__pycache__" / "foo.pyc", root)
    assert is_tool_output(root / "src" / "__pycache__" / "bar.pyc", root)

    # Regular source files should not be detected as tool output
    assert not is_tool_output(root / "src" / "main.py", root)
    assert not is_tool_output(root / "tests" / "test_foo.py", root)

    # .git should be excluded elsewhere but not by this function
    # (this only checks TOOL_OUTPUT_DIRS, not .git)
    assert not is_tool_output(root / ".git" / "config", root)


def test_strip_python_docstrings_module_level():
    """Test stripping module-level docstrings."""
    source = '''"""
This is a module docstring.
It spans multiple lines.
"""

def foo():
    return 42
'''
    result = strip_python_docstrings(source)
    assert '"""' not in result or result.count('"""') < 2
    assert "def foo():" in result
    assert "return 42" in result


def test_strip_python_docstrings_function():
    """Test stripping function docstrings."""
    source = '''def foo():
    """This is a function docstring."""
    return 42

def bar():
    """
    Multi-line function docstring.
    With multiple lines.
    """
    return "hello"
'''
    result = strip_python_docstrings(source)
    assert "def foo():" in result
    assert "def bar():" in result
    assert "return 42" in result
    assert 'return "hello"' in result
    # The docstrings should be gone or replaced
    # Check that the docstring text doesn't appear
    assert "This is a function docstring" not in result
    assert "Multi-line function docstring" not in result


def test_strip_python_docstrings_class():
    """Test stripping class docstrings."""
    source = '''class MyClass:
    """This is a class docstring."""

    def method(self):
        """Method docstring."""
        pass
'''
    result = strip_python_docstrings(source)
    assert "class MyClass:" in result
    assert "def method(self):" in result
    assert "pass" in result
    # Docstrings should be stripped
    assert "This is a class docstring" not in result
    assert "Method docstring" not in result


def test_strip_python_docstrings_preserves_strings():
    """Test that non-docstring string literals are preserved."""
    source = '''def foo():
    """Docstring to remove."""
    x = "This is a regular string"
    y = "Another regular string"
    return x + y
'''
    result = strip_python_docstrings(source)
    assert "def foo():" in result
    # The immediate-after-def docstring must be gone.
    assert "Docstring to remove" not in result
    # The assigned (non-docstring) string literals must REMAIN.
    assert '"This is a regular string"' in result
    assert '"Another regular string"' in result
    assert "return x + y" in result


def test_strip_python_docstrings_async_def():
    """Test stripping docstrings from async functions."""
    source = '''async def fetch(url):
    """Fetch the URL asynchronously."""
    return await client.get(url)
'''
    result = strip_python_docstrings(source)
    assert "async def fetch(url):" in result
    assert "return await client.get(url)" in result
    assert "Fetch the URL asynchronously" not in result


def test_strip_python_docstrings_after_blank_and_comment_lines():
    """Test stripping docstrings preceded by blank/comment lines after header."""
    source = '''def foo():

    # a leading comment
    """Docstring after blank + comment."""
    return 1

class Bar:

    """Class docstring after a blank line."""
    x = 1
'''
    result = strip_python_docstrings(source)
    assert "def foo():" in result
    assert "class Bar:" in result
    assert "return 1" in result
    assert "x = 1" in result
    # The interleaved comment is preserved; docstrings are gone.
    assert "# a leading comment" in result
    assert "Docstring after blank + comment" not in result
    assert "Class docstring after a blank line" not in result


def test_strip_python_docstrings_prefixed():
    """Test stripping prefixed string docstrings (r/b/f/u variants)."""
    source = (
        'def raw_doc():\n'
        '    r"""Raw docstring with regex."""\n'
        '    return 1\n'
        '\n'
        'def fstr_doc():\n'
        '    f"""F-string docstring here."""\n'
        '    return 2\n'
        '\n'
        'def bytes_doc():\n'
        '    b"""Bytes docstring."""\n'
        '    return 3\n'
        '\n'
        'def single_quote_raw():\n'
        "    r'''Single-quote raw docstring.'''\n"
        '    return 4\n'
    )
    result = strip_python_docstrings(source)
    assert "def raw_doc():" in result
    assert "def fstr_doc():" in result
    assert "def bytes_doc():" in result
    assert "def single_quote_raw():" in result
    assert "return 1" in result
    assert "return 2" in result
    assert "return 3" in result
    assert "return 4" in result
    # All prefixed docstrings must be stripped.
    assert "Raw docstring" not in result
    assert "F-string docstring" not in result
    assert "Bytes docstring" not in result
    assert "Single-quote raw docstring" not in result


def test_strip_python_docstrings_prefixed_module():
    """Test stripping a prefixed module-level docstring."""
    source = (
        'r"""Raw module docstring."""\n'
        '\n'
        'import os\n'
    )
    result = strip_python_docstrings(source)
    assert "import os" in result
    assert "Raw module docstring" not in result


def test_strip_python_docstrings_empty():
    """Test stripping from empty or minimal code."""
    assert strip_python_docstrings("") == ""
    assert strip_python_docstrings("# just a comment") == "# just a comment"


def test_strip_python_docstrings_complex():
    """Test a more complex real-world example."""
    source = '''#!/usr/bin/env python
"""
Module docstring for example.py.
This should be stripped.
"""

import os
import sys

class Processor:
    """
    A processor class.
    This docstring should be removed.
    """

    def process(self, data):
        """
        Process the data.
        This should also be removed.
        """
        # This is a comment, keep it
        result = data.upper()
        return result

def main():
    """Main entry point."""
    p = Processor()
    print(p.process("hello"))

if __name__ == "__main__":
    main()
'''
    result = strip_python_docstrings(source)

    # Check structure is preserved
    assert "#!/usr/bin/env python" in result
    assert "import os" in result
    assert "import sys" in result
    assert "class Processor:" in result
    assert "def process(self, data):" in result
    assert "def main():" in result
    assert 'if __name__ == "__main__":' in result

    # Check code is preserved
    assert "result = data.upper()" in result
    assert "return result" in result
    assert "# This is a comment, keep it" in result

    # Check docstrings are gone
    assert "Module docstring for example.py" not in result
    assert "A processor class" not in result
    assert "Process the data" not in result
    assert "Main entry point" not in result
