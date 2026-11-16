"""Entry point of the fixture repo."""

from demo.core import format_name, parse_arguments


def main(argv):
    """Parse argv and print each normalised name."""
    arguments = parse_arguments(argv)
    return [format_name(a.name) for a in arguments]
