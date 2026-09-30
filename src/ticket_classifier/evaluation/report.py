"""Side-by-side comparison of a Baseline and a Transformer model version.

Only the metrics already recorded in the registry for the ``test`` split are used;
models are never re-evaluated here.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ticket_classifier.errors import PipelineError
from ticket_classifier.evaluation.metrics import TargetMetrics
from ticket_classifier.labels import TARGETS
from ticket_classifier.registry import ModelVersion

EVALUATION_SPLIT = "test"
REPORT_MARKDOWN = "comparison.md"
REPORT_JSON = "comparison.json"

# Premise LAC-05: category must gain at least 5 pp, priority must not drop.
CATEGORY_MIN_GAIN = 0.05
PRIORITY_MIN_GAIN = 0.0
_TOLERANCE = 1e-9

_MODELS: tuple[tuple[str, str], ...] = (("baseline", "Baseline"), ("transformer", "Transformer"))

Verdict = Literal["PASS", "FAIL"]


@dataclass
class ComparisonReport:
    """Test-split metrics of both models per target, Macro F1 deltas and verdict.

    ``metrics[target][model]`` holds ``TargetMetrics.to_dict()`` output, with
    ``model`` in ``("baseline", "transformer")``.
    """

    baseline_id: str
    transformer_id: str
    metrics: dict[str, dict[str, dict[str, Any]]]
    macro_f1_delta_pp: dict[str, float]
    verdict: Verdict


def _test_metrics(version: ModelVersion, target: str) -> TargetMetrics:
    data = version.metrics.get(EVALUATION_SPLIT, {}).get(target)
    if not isinstance(data, Mapping):
        raise PipelineError(
            f"Model version {version.version_id} has no {EVALUATION_SPLIT} metrics for {target}."
        )
    try:
        return TargetMetrics.from_dict(data)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PipelineError(
            f"Model version {version.version_id} has invalid {EVALUATION_SPLIT} metrics "
            f"for {target}."
        ) from exc


def _verdict(baseline: dict[str, TargetMetrics], transformer: dict[str, TargetMetrics]) -> Verdict:
    category_ok = (
        transformer["category"].macro_f1
        >= baseline["category"].macro_f1 + CATEGORY_MIN_GAIN - _TOLERANCE
    )
    priority_ok = (
        transformer["priority"].macro_f1
        >= baseline["priority"].macro_f1 + PRIORITY_MIN_GAIN - _TOLERANCE
    )
    return "PASS" if category_ok and priority_ok else "FAIL"


def build_comparison(baseline: ModelVersion, transformer: ModelVersion) -> ComparisonReport:
    """Compare the recorded test metrics of ``baseline`` and ``transformer``."""
    if baseline.splits_version != transformer.splits_version:
        raise PipelineError("Model versions were evaluated on different splits.")
    base = {target: _test_metrics(baseline, target) for target in TARGETS}
    trans = {target: _test_metrics(transformer, target) for target in TARGETS}
    return ComparisonReport(
        baseline_id=baseline.version_id,
        transformer_id=transformer.version_id,
        metrics={
            target: {"baseline": base[target].to_dict(), "transformer": trans[target].to_dict()}
            for target in TARGETS
        },
        macro_f1_delta_pp={
            target: round((trans[target].macro_f1 - base[target].macro_f1) * 100, 2)
            for target in TARGETS
        },
        verdict=_verdict(base, trans),
    )


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _target_section(target: str, report: ComparisonReport) -> list[str]:
    metrics = {
        model: TargetMetrics.from_dict(report.metrics[target][model]) for model, _ in _MODELS
    }
    base, trans = metrics["baseline"], metrics["transformer"]
    lines = [f"## Target: {target}", ""]
    lines += _table(
        ["Metric", "Baseline", "Transformer", "Delta (pp)"],
        [
            [
                "Accuracy",
                _fmt(base.accuracy),
                _fmt(trans.accuracy),
                f"{round((trans.accuracy - base.accuracy) * 100, 2):+.2f}",
            ],
            [
                "Macro F1",
                _fmt(base.macro_f1),
                _fmt(trans.macro_f1),
                f"{report.macro_f1_delta_pp[target]:+.2f}",
            ],
        ],
    )
    lines += ["", "### Per-class metrics", ""]
    per_class_rows = []
    for label in base.labels:
        b = base.per_class[label]
        t = trans.per_class.get(label)
        t_values = [_fmt(t.precision), _fmt(t.recall), _fmt(t.f1)] if t else ["-", "-", "-"]
        per_class_rows.append(
            [label, _fmt(b.precision), _fmt(b.recall), _fmt(b.f1), *t_values, str(b.support)]
        )
    lines += _table(
        [
            "Class",
            "Baseline precision",
            "Baseline recall",
            "Baseline F1",
            "Transformer precision",
            "Transformer recall",
            "Transformer F1",
            "Support",
        ],
        per_class_rows,
    )
    for model, title in _MODELS:
        matrix = metrics[model]
        lines += ["", f"### Confusion matrix ({target}, {title})", ""]
        lines += _table(
            ["True \\ Predicted", *matrix.labels],
            [
                [label, *(str(cell) for cell in row)]
                for label, row in zip(matrix.labels, matrix.confusion_matrix, strict=True)
            ],
        )
    lines.append("")
    return lines


def render_markdown(report: ComparisonReport) -> str:
    """Render the comparison as a Markdown document."""
    lines = [
        "# Baseline vs Transformer comparison",
        "",
        f"- Baseline: `{report.baseline_id}`",
        f"- Transformer: `{report.transformer_id}`",
        f"- Split: `{EVALUATION_SPLIT}`",
        "",
        f"Verdict: {report.verdict}",
        "",
        "Rule: PASS if Transformer category Macro F1 >= Baseline + 5 pp and Transformer "
        "priority Macro F1 >= Baseline; FAIL otherwise.",
        "",
    ]
    for target in TARGETS:
        lines += _target_section(target, report)
    return "\n".join(lines)


def write_report(report: ComparisonReport, out_dir: Path) -> tuple[Path, Path]:
    """Write ``comparison.md`` and ``comparison.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    md_path = out_dir / REPORT_MARKDOWN
    json_path = out_dir / REPORT_JSON
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(report), encoding="utf-8")
        json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    except OSError as exc:
        raise PipelineError("Cannot write the comparison report.") from exc
    return md_path, json_path
