"""Command-line entry point for ``ticket-classifier``.

Each subcommand is registered by a ``_add_<name>_command(subparsers)`` function that
calls ``parser.set_defaults(handler=<function(args) -> int>)``.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ticket_classifier.errors import PipelineError
from ticket_classifier.registry import ModelRegistry

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
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _add_list_versions_command(subparsers)
    return parser


_LIST_VERSIONS_HEADER = (
    "version_id",
    "kind",
    "test_category_macro_f1",
    "test_priority_macro_f1",
    "created_at",
    "promoted",
)


def _add_list_versions_command(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser("list-versions", help="List registered model versions.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing registry.json (default: artifacts).",
    )
    parser.set_defaults(handler=_run_list_versions)


def _format_f1(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _run_list_versions(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.artifacts_dir)
    rows: list[tuple[str, ...]] = [_LIST_VERSIONS_HEADER]
    for version in registry.list_versions():
        rows.append(
            (
                version.version_id,
                version.kind,
                _format_f1(version.test_macro_f1("category")),
                _format_f1(version.test_macro_f1("priority")),
                version.created_at,
                "yes" if version.status == "promoted" else "no",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(_LIST_VERSIONS_HEADER))]
    for row in rows:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )
    return 0


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
