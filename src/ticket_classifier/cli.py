"""Command-line entry point for ``ticket-classifier``.

Each subcommand is registered by a ``_add_<name>_command(subparsers)`` function that
calls ``parser.set_defaults(handler=<function(args) -> int>)``.
"""

import argparse
import sqlite3
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from ticket_classifier.data.prepare import prepare_dataset
from ticket_classifier.errors import PipelineError
from ticket_classifier.pipeline_config import load_pipeline_config
from ticket_classifier.registry import ModelRegistry
from ticket_classifier.settings import get_settings
from ticket_classifier.storage.database import connect, init_schema, utc_now
from ticket_classifier.storage.purge import PURGE_MESSAGE, purge_expired_predictions

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
    _add_prepare_command(subparsers)
    _add_purge_command(subparsers)
    _add_train_baseline_command(subparsers)
    _add_train_transformer_command(subparsers)
    return parser


def _add_prepare_command(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser(
        "prepare", help="Prepare the train/validation/test splits from the source dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.toml"),
        help="Pipeline configuration file (default: configs/pipeline.toml).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("configs/label_mapping.toml"),
        help="Label mapping table (default: configs/label_mapping.toml).",
    )
    parser.set_defaults(handler=_run_prepare)


def _run_prepare(args: argparse.Namespace) -> int:
    config = load_pipeline_config(args.config)
    report = prepare_dataset(config, args.mapping)
    print(
        f"Prepared splits in {config.data.processed_dir} "
        f"(splits_version {report['splits_version']})."
    )
    return 0


def _add_train_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.toml"),
        help="Pipeline configuration file (default: configs/pipeline.toml).",
    )


def _add_train_baseline_command(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser(
        "train-baseline", help="Train, evaluate and register a Baseline model version."
    )
    _add_train_config_argument(parser)
    parser.set_defaults(handler=_run_train_baseline)


def _add_train_transformer_command(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser(
        "train-transformer", help="Train, evaluate and register a Transformer model version."
    )
    _add_train_config_argument(parser)
    parser.set_defaults(handler=_run_train_transformer)


def _run_train_baseline(args: argparse.Namespace) -> int:
    return _run_training("baseline", args.config)


def _run_train_transformer(args: argparse.Namespace) -> int:
    return _run_training("transformer", args.config)


def _run_training(kind: Literal["baseline", "transformer"], config_path: Path) -> int:
    # Imported lazily: training pulls in torch, transformers and mlflow.
    from ticket_classifier.training import train_and_register

    config = load_pipeline_config(config_path)
    outcome = train_and_register(kind, config, get_settings())
    promoted = "yes" if outcome.promoted else "no"
    print(
        f"Registered model version '{outcome.version.version_id}' ({kind}). Promoted: {promoted}."
    )
    return 0


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


def _add_purge_command(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser(
        "purge",
        help="Delete stored predictions without feedback older than the retention period.",
    )
    parser.set_defaults(handler=_run_purge)


def _run_purge(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        conn = connect(settings.db_path)
        try:
            init_schema(conn)
            count = purge_expired_predictions(conn, utc_now(), settings.retention_days)
        finally:
            conn.close()
    except sqlite3.Error:
        print("Purge failed: the prediction database could not be accessed.", file=sys.stderr)
        return _PIPELINE_ERROR
    print(PURGE_MESSAGE.format(count=count, days=settings.retention_days))
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
