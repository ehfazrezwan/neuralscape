"""Sample Python file for oracle testing."""


def helper_function(x):
    """A simple helper function."""
    return x * 2


class Calculator:
    """A simple calculator class."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def multiply(self, a, b):
        """Multiply two numbers."""
        result = helper_function(a)
        return result * b


def main():
    """Main entry point."""
    calc = Calculator()
    result = calc.add(1, 2)
    calc.multiply(3, 4)
    return result
