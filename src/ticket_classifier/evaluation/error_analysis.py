"""Error analysis of a model version on the ``test`` split (MODEL-10, MODEL-11).

An error is a ticket whose predicted ``category`` or ``priority`` differs from the
true label. ``needs_review`` follows LAC-22: the category or the priority
confidence is below the review threshold.
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ticket_classifier.data.splits import EmptySplitError
from ticket_classifier.errors import PipelineError
from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.base import TicketClassifier, top_label

EVALUATION_SPLIT = "test"
TOP_CONFUSIONS = 5
REPORT_PREFIX = "error_analysis_"

_SAFE_VERSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass
class ErrorAnalysis:
    """Misclassified tickets, top category confusions and review statistics."""

    version_id: str
    threshold: float
    errors: list[dict[str, Any]]
    top_category_confusions: list[tuple[str, str, int]]
    needs_review_fraction: float
    category_accuracy_needs_review: float | None
    category_accuracy_confident: float | None


def _accuracy(hits: list[bool]) -> float | None:
    return sum(hits) / len(hits) if hits else None


def _top_confusions(pairs: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
    counts = Counter(pairs)
    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], CATEGORIES.index(item[0][0]), CATEGORIES.index(item[0][1])),
    )
    return [(true, predicted, count) for (true, predicted), count in ordered[:TOP_CONFUSIONS]]


def analyze_errors(
    classifier: TicketClassifier, version_id: str, test: pd.DataFrame, threshold: float
) -> ErrorAnalysis:
    """Analyze the predictions of ``classifier`` on the ``test`` frame."""
    if test.empty:
        raise EmptySplitError(EVALUATION_SPLIT)
    titles = [str(value) for value in test["title"].tolist()]
    descriptions = [str(value) for value in test["description"].tolist()]
    true_categories = [str(value) for value in test["category"].tolist()]
    true_priorities = [str(value) for value in test["priority"].tolist()]
    predictions = classifier.predict_proba(titles, descriptions)
    if len(predictions) != len(titles):
        raise PipelineError("The classifier returned an unexpected number of predictions.")

    errors: list[dict[str, Any]] = []
    confusions: list[tuple[str, str]] = []
    hits_needs_review: list[bool] = []
    hits_confident: list[bool] = []
    for index, probs in enumerate(predictions):
        category, category_confidence = top_label(probs.category, CATEGORIES)
        priority, priority_confidence = top_label(probs.priority, PRIORITIES)
        category_hit = category == true_categories[index]
        priority_hit = priority == true_priorities[index]
        needs_review = category_confidence < threshold or priority_confidence < threshold
        (hits_needs_review if needs_review else hits_confident).append(category_hit)
        if not category_hit:
            confusions.append((true_categories[index], category))
        if not (category_hit and priority_hit):
            errors.append(
                {
                    "title": titles[index],
                    "description": descriptions[index],
                    "true_category": true_categories[index],
                    "predicted_category": category,
                    "category_confidence": category_confidence,
                    "true_priority": true_priorities[index],
                    "predicted_priority": priority,
                    "priority_confidence": priority_confidence,
                }
            )

    return ErrorAnalysis(
        version_id=version_id,
        threshold=threshold,
        errors=errors,
        top_category_confusions=_top_confusions(confusions),
        needs_review_fraction=len(hits_needs_review) / len(predictions),
        category_accuracy_needs_review=_accuracy(hits_needs_review),
        category_accuracy_confident=_accuracy(hits_confident),
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(analysis: ErrorAnalysis) -> str:
    """Render the error analysis as a Markdown document."""
    lines = [
        f"# Error analysis: `{analysis.version_id}`",
        "",
        f"- Split: `{EVALUATION_SPLIT}`",
        f"- Review threshold: {analysis.threshold}",
        f"- Errors: {len(analysis.errors)}",
        "",
        "## Low confidence (needs_review)",
        "",
    ]
    lines += _table(
        ["Metric", "Value"],
        [
            ["needs_review fraction", _fmt(analysis.needs_review_fraction)],
            [
                "Category accuracy (needs_review = true)",
                _fmt(analysis.category_accuracy_needs_review),
            ],
            [
                "Category accuracy (needs_review = false)",
                _fmt(analysis.category_accuracy_confident),
            ],
        ],
    )
    lines += ["", f"## Top {TOP_CONFUSIONS} category confusions", ""]
    lines += _table(
        ["True", "Predicted", "Count"],
        [
            [true, predicted, str(count)]
            for true, predicted, count in analysis.top_category_confusions
        ],
    )
    lines += ["", "## Misclassified tickets", ""]
    lines += _table(
        [
            "Title",
            "Description",
            "True category",
            "Predicted category",
            "Category confidence",
            "True priority",
            "Predicted priority",
            "Priority confidence",
        ],
        [
            [
                _cell(error["title"]),
                _cell(error["description"]),
                error["true_category"],
                error["predicted_category"],
                _fmt(error["category_confidence"]),
                error["true_priority"],
                error["predicted_priority"],
                _fmt(error["priority_confidence"]),
            ]
            for error in analysis.errors
        ],
    )
    lines.append("")
    return "\n".join(lines)


def write_error_analysis(analysis: ErrorAnalysis, out_dir: Path) -> tuple[Path, Path]:
    """Write ``error_analysis_<id>.md`` and ``error_analysis_<id>.json`` into ``out_dir``."""
    if not _SAFE_VERSION_ID.match(analysis.version_id):
        raise PipelineError("Invalid model version id.")
    out_dir = Path(out_dir)
    md_path = out_dir / f"{REPORT_PREFIX}{analysis.version_id}.md"
    json_path = out_dir / f"{REPORT_PREFIX}{analysis.version_id}.json"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(analysis), encoding="utf-8")
        json_path.write_text(
            json.dumps(asdict(analysis), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        raise PipelineError("Cannot write the error analysis report.") from exc
    return md_path, json_path
