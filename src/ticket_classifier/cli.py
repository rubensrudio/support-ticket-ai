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
from ticket_classifier.registry import ModelRegistry, ModelVersion
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
    _add_analyze_errors_command(subparsers)
    _add_compare_command(subparsers)
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


def _add_compare_command(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser(
        "compare", help="Compare a Baseline and a Transformer model version on the test split."
    )
    parser.add_argument(
        "--baseline",
        metavar="ID",
        default=None,
        help="Baseline version id (default: most recent baseline version).",
    )
    parser.add_argument(
        "--transformer",
        metavar="ID",
        default=None,
        help="Transformer version id (default: most recent transformer version).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing registry.json; the report goes to <dir>/reports "
        "(default: artifacts).",
    )
    parser.set_defaults(handler=_run_compare)


def _resolve_version(registry: ModelRegistry, kind: str, version_id: str | None) -> ModelVersion:
    if version_id is None:
        version = registry.latest(kind)
        if version is None:
            raise PipelineError(f"No '{kind}' model version registered. Train it first.")
        return version
    version = registry.get(version_id)
    if version.kind != kind:
        raise PipelineError(f"Model version {version_id} is not a '{kind}' version.")
    return version


def _run_compare(args: argparse.Namespace) -> int:
    from ticket_classifier.evaluation.report import build_comparison, write_report

    registry = ModelRegistry(args.artifacts_dir)
    baseline = _resolve_version(registry, "baseline", args.baseline)
    transformer = _resolve_version(registry, "transformer", args.transformer)
    report = build_comparison(baseline, transformer)
    md_path, json_path = write_report(report, args.artifacts_dir / "reports")
    rows: list[tuple[str, ...]] = [
        ("target", "baseline_macro_f1", "transformer_macro_f1", "delta_pp")
    ]
    for target, delta in report.macro_f1_delta_pp.items():
        rows.append(
            (
                target,
                _format_f1(baseline.test_macro_f1(target)),
                _format_f1(transformer.test_macro_f1(target)),
                f"{delta:+.2f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    print(f"Baseline: {report.baseline_id}")
    print(f"Transformer: {report.transformer_id}")
    for row in rows:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )
    print(f"Verdict: {report.verdict}")
    print(f"Report: {md_path} and {json_path}")
    return 0


_NO_PROMOTED_MESSAGE = "No promoted model version. Train a transformer first."


def _add_analyze_errors_command(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser(
        "analyze-errors", help="Analyze the errors of a model version on the test split."
    )
    parser.add_argument(
        "--version",
        metavar="ID",
        default=None,
        help="Model version id (default: the promoted version).",
    )
    _add_train_config_argument(parser)
    parser.set_defaults(handler=_run_analyze_errors)


def _run_analyze_errors(args: argparse.Namespace) -> int:
    # Imported lazily: loading a classifier pulls in torch and transformers.
    from ticket_classifier.data.splits import load_split
    from ticket_classifier.evaluation.error_analysis import analyze_errors, write_error_analysis
    from ticket_classifier.models.loader import load_classifier

    settings = get_settings()
    registry = ModelRegistry(settings.artifacts_dir)
    if args.version is None:
        version = registry.get_promoted()
        if version is None:
            raise PipelineError(_NO_PROMOTED_MESSAGE)
    else:
        version = registry.get(args.version)
    config = load_pipeline_config(args.config)
    test = load_split(config.data.processed_dir, "test")
    classifier = load_classifier(version, settings.artifacts_dir)
    analysis = analyze_errors(classifier, version.version_id, test, settings.review_threshold)
    md_path, json_path = write_error_analysis(analysis, settings.artifacts_dir / "reports")
    print(f"Model version: {version.version_id}")
    print(f"Errors: {len(analysis.errors)} of {len(test)} test tickets")
    print(f"Review threshold: {analysis.threshold}")
    print(f"needs_review fraction: {analysis.needs_review_fraction:.4f}")
    print(f"Report: {md_path} and {json_path}")
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
