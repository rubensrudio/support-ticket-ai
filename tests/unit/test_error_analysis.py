import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from ticket_classifier.errors import PipelineError
from ticket_classifier.evaluation.error_analysis import (
    ErrorAnalysis,
    analyze_errors,
    write_error_analysis,
)
from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.base import ClassProbabilities


def _probs(labels: Sequence[str], top: str, confidence: float) -> dict[str, float]:
    rest = (1.0 - confidence) / (len(labels) - 1)
    return {label: (confidence if label == top else rest) for label in labels}


# (true category, predicted category, category confidence,
#  true priority, predicted priority, priority confidence)
_ROWS: list[tuple[str, str, float, str, str, float]] = [
    ("access", "access", 0.9, "low", "low", 0.9),  # correct, confident
    ("access", "billing", 0.8, "low", "low", 0.9),  # category error, confident
    ("access", "billing", 0.5, "low", "low", 0.9),  # category error, needs review
    ("bug", "other", 0.7, "high", "high", 0.7),  # category error, confident
    ("bug", "other", 0.7, "high", "high", 0.7),  # category error, confident
    ("bug", "other", 0.7, "high", "high", 0.7),  # category error, confident
    ("infrastructure", "access", 0.9, "medium", "medium", 0.9),  # category error
    ("billing", "access", 0.9, "medium", "medium", 0.9),  # category error
    ("other", "bug", 0.9, "medium", "medium", 0.9),  # category error
    ("other", "infrastructure", 0.9, "medium", "medium", 0.9),  # category error
    ("billing", "billing", 0.9, "medium", "high", 0.4),  # priority error, needs review
    ("other", "other", 0.55, "low", "low", 0.9),  # correct, needs review
]


class FakeClassifier:
    kind = "baseline"

    def __init__(self, outputs: dict[str, ClassProbabilities]) -> None:
        self._outputs = outputs

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        return [self._outputs[title] for title in titles]

    def save(self, directory: Path) -> None:  # pragma: no cover - not used
        raise NotImplementedError


def _fixture() -> tuple[FakeClassifier, pd.DataFrame]:
    outputs: dict[str, ClassProbabilities] = {}
    records = []
    for index, (cat, cat_pred, cat_conf, pri, pri_pred, pri_conf) in enumerate(_ROWS):
        title = f"ticket {index}"
        outputs[title] = ClassProbabilities(
            category=_probs(CATEGORIES, cat_pred, cat_conf),
            priority=_probs(PRIORITIES, pri_pred, pri_conf),
        )
        records.append(
            {
                "title": title,
                "description": f"description {index}",
                "category": cat,
                "priority": pri,
            }
        )
    return FakeClassifier(outputs), pd.DataFrame(records)


def test_model10_errors_list_has_expected_items_and_fields() -> None:
    classifier, test = _fixture()

    analysis = analyze_errors(classifier, "baseline-v1", test, threshold=0.6)

    assert analysis.version_id == "baseline-v1"
    assert analysis.threshold == 0.6
    assert len(analysis.errors) == 10
    first = analysis.errors[0]
    assert first == {
        "title": "ticket 1",
        "description": "description 1",
        "true_category": "access",
        "predicted_category": "billing",
        "category_confidence": pytest.approx(0.8),
        "true_priority": "low",
        "predicted_priority": "low",
        "priority_confidence": pytest.approx(0.9),
    }
    priority_error = next(e for e in analysis.errors if e["title"] == "ticket 10")
    assert priority_error["true_priority"] == "medium"
    assert priority_error["predicted_priority"] == "high"
    assert priority_error["priority_confidence"] == pytest.approx(0.4)
    titles = {e["title"] for e in analysis.errors}
    assert "ticket 0" not in titles
    assert "ticket 11" not in titles


def test_model10_top_confusions_sorted_by_count_then_label_order() -> None:
    classifier, test = _fixture()

    analysis = analyze_errors(classifier, "baseline-v1", test, threshold=0.6)

    # Counts: (bug, other)=3, (access, billing)=2, then four pairs with count 1
    # ordered by CT-1 position of the true label, then of the predicted label.
    assert analysis.top_category_confusions == [
        ("bug", "other", 3),
        ("access", "billing", 2),
        ("infrastructure", "access", 1),
        ("billing", "access", 1),
        ("other", "infrastructure", 1),
    ]


def test_model11_needs_review_fraction_and_accuracy_per_group() -> None:
    classifier, test = _fixture()

    analysis = analyze_errors(classifier, "baseline-v1", test, threshold=0.6)

    # needs_review rows: 2 (cat 0.5), 10 (pri 0.4), 11 (cat 0.55) -> 3 of 12.
    assert analysis.needs_review_fraction == pytest.approx(3 / 12)
    # Category correct among needs_review: rows 10 and 11 -> 2/3.
    assert analysis.category_accuracy_needs_review == pytest.approx(2 / 3)
    # Category correct among confident rows (0,1,3..9): only row 0 -> 1/9.
    assert analysis.category_accuracy_confident == pytest.approx(1 / 9)


def test_model11_threshold_zero_means_nobody_needs_review() -> None:
    classifier, test = _fixture()

    analysis = analyze_errors(classifier, "baseline-v1", test, threshold=0.0)

    assert analysis.needs_review_fraction == 0.0
    assert analysis.category_accuracy_needs_review is None
    assert analysis.category_accuracy_confident == pytest.approx(3 / 12)


def test_model11_threshold_one_means_everybody_needs_review() -> None:
    classifier, test = _fixture()

    analysis = analyze_errors(classifier, "baseline-v1", test, threshold=1.0)

    assert analysis.needs_review_fraction == 1.0
    assert analysis.category_accuracy_needs_review == pytest.approx(3 / 12)
    assert analysis.category_accuracy_confident is None


def test_model10_empty_test_split_raises_pipeline_error() -> None:
    classifier, test = _fixture()

    with pytest.raises(PipelineError):
        analyze_errors(classifier, "baseline-v1", test.iloc[0:0], threshold=0.6)


def test_model10_write_error_analysis_creates_markdown_and_json(tmp_path: Path) -> None:
    classifier, test = _fixture()
    analysis = analyze_errors(classifier, "baseline-v1", test, threshold=0.6)

    md_path, json_path = write_error_analysis(analysis, tmp_path / "reports")

    assert md_path == tmp_path / "reports" / "error_analysis_baseline-v1.md"
    assert json_path == tmp_path / "reports" / "error_analysis_baseline-v1.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["version_id"] == "baseline-v1"
    assert data["threshold"] == 0.6
    assert len(data["errors"]) == 10
    assert data["top_category_confusions"][0] == ["bug", "other", 3]
    assert data["needs_review_fraction"] == pytest.approx(0.25)
    markdown = md_path.read_text(encoding="utf-8")
    assert "baseline-v1" in markdown
    assert "| bug | other | 3 |" in markdown
    assert "ticket 1" in markdown


def test_model10_write_error_analysis_rejects_unsafe_version_id(tmp_path: Path) -> None:
    analysis = ErrorAnalysis(
        version_id="../escape",
        threshold=0.6,
        errors=[],
        top_category_confusions=[],
        needs_review_fraction=0.0,
        category_accuracy_needs_review=None,
        category_accuracy_confident=None,
    )

    with pytest.raises(PipelineError):
        write_error_analysis(analysis, tmp_path)
    assert list(tmp_path.iterdir()) == []
