"""Argument handling for the code-engine integration fixture."""


class Argument:
    """A parsed command line argument."""

    def __init__(self, name):
        self.name = name

    def render(self):
        """Render the argument for display."""
        return format_name(self.name)


def format_name(name):
    """Normalise an argument name for display."""
    return name.strip().lower()


def parse_arguments(raw):
    """Parse raw strings into Argument objects."""
    return [Argument(item) for item in raw]
