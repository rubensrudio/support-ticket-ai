"""Command-line entry point for ``ticket-classifier``.

Each subcommand is registered by a ``_add_<name>_command(subparsers)`` function that
calls ``parser.set_defaults(handler=<function(args) -> int>)``.
"""

import argparse
import sys
from collections.abc import Callable, Sequence

from ticket_classifier.errors import PipelineError

_PROG = "ticket-classifier"
_USAGE_ERROR = 2
_PIPELINE_ERROR = 1

Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with all registered subcommands."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Support ticket classification pipeline and REST API.",
    )
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_usage(sys.stderr)
        return _USAGE_ERROR
    try:
        return handler(args)
    except PipelineError as exc:
        print(exc.message, file=sys.stderr)
        return _PIPELINE_ERROR
