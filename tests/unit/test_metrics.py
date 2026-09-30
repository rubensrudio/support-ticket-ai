from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
from sklearn.metrics import f1_score

from ticket_classifier.evaluation import metrics as metrics_module
from ticket_classifier.evaluation.metrics import (
    ClassMetrics,
    TargetMetrics,
    compute_target_metrics,
    evaluate_classifier,
)
from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.base import ClassProbabilities, TicketClassifier

Y_TRUE = ["a", "a", "b", "c"]
Y_PRED = ["a", "b", "b", "c"]
LABELS = ["a", "b", "c"]


def test_model03_known_case_accuracy_macro_f1_and_confusion_matrix() -> None:
    result = compute_target_metrics(Y_TRUE, Y_PRED, LABELS)

    assert result.labels == LABELS
    assert result.accuracy == pytest.approx(0.75)
    assert result.macro_f1 == pytest.approx(
        f1_score(Y_TRUE, Y_PRED, labels=LABELS, average="macro", zero_division=0)
    )
    assert result.confusion_matrix == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_model03_per_class_precision_recall_f1_support() -> None:
    result = compute_target_metrics(Y_TRUE, Y_PRED, LABELS)

    assert list(result.per_class) == LABELS
    assert result.per_class["a"] == ClassMetrics(
        precision=1.0, recall=0.5, f1=pytest.approx(2 / 3), support=2
    )
    assert result.per_class["b"].precision == pytest.approx(0.5)
    assert result.per_class["b"].recall == pytest.approx(1.0)
    assert result.per_class["c"].support == 1


def test_model03_label_without_support_is_reported_with_zero_f1() -> None:
    result = compute_target_metrics(Y_TRUE, Y_PRED, ["a", "b", "c", "d"])

    assert result.per_class["d"].support == 0
    assert result.per_class["d"].f1 == 0.0
    assert result.per_class["d"].precision == 0.0
    assert result.per_class["d"].recall == 0.0
    assert len(result.confusion_matrix) == 4
    assert result.confusion_matrix[3] == [0, 0, 0, 0]


def test_model03_metrics_values_are_builtin_types() -> None:
    result = compute_target_metrics(Y_TRUE, Y_PRED, LABELS)

    assert type(result.accuracy) is float
    assert type(result.macro_f1) is float
    assert type(result.per_class["a"].support) is int
    assert type(result.per_class["a"].f1) is float
    assert all(type(cell) is int for row in result.confusion_matrix for cell in row)


def test_model03_to_dict_from_dict_round_trip() -> None:
    result = compute_target_metrics(Y_TRUE, Y_PRED, ["a", "b", "c", "d"])

    data = result.to_dict()

    assert isinstance(data["per_class"]["a"], dict)
    assert TargetMetrics.from_dict(data) == result


def test_model03_compute_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_target_metrics(["a"], ["a", "b"], LABELS)


class _FakeClassifier:
    kind: Literal["baseline", "transformer"] = "baseline"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        assert len(titles) == len(descriptions)
        self.batch_sizes.append(len(titles))
        result = []
        for title in titles:
            category = {label: 0.0 for label in CATEGORIES}
            category[title] = 1.0
            priority = {label: 1 / len(PRIORITIES) for label in PRIORITIES}
            result.append(ClassProbabilities(category=category, priority=priority))
        return result

    def save(self, directory: Path) -> None:
        return None


def _frame(size: int) -> pd.DataFrame:
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(size)]
    return pd.DataFrame(
        {
            "title": categories,
            "description": ["body"] * size,
            "category": categories,
            "priority": ["medium"] * size,
        }
    )


def test_model03_evaluate_classifier_returns_both_targets() -> None:
    classifier = _FakeClassifier()
    assert isinstance(classifier, TicketClassifier)

    result = evaluate_classifier(classifier, _frame(10))

    assert set(result) == {"category", "priority"}
    assert result["category"].labels == list(CATEGORIES)
    assert result["category"].accuracy == pytest.approx(1.0)
    assert result["priority"].labels == list(PRIORITIES)
    # Uniform priority probabilities tie; top_label picks the first label ("low").
    assert result["priority"].accuracy == pytest.approx(0.0)
    assert result["priority"].per_class["medium"].support == 10


def test_model03_evaluate_classifier_uses_batches_of_64() -> None:
    classifier = _FakeClassifier()

    evaluate_classifier(classifier, _frame(130))

    assert classifier.batch_sizes == [64, 64, 2]
    assert metrics_module.BATCH_SIZE == 64
