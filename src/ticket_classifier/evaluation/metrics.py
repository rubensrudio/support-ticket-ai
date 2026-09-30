"""Classification metrics per target (category, priority).

Metrics always use the full, fixed label list of each target, so labels with
no support in the evaluated split still appear (with zeroed scores).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.base import TicketClassifier, top_label

BATCH_SIZE = 64


@dataclass
class ClassMetrics:
    """Precision, recall, F1 and support of a single label."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class TargetMetrics:
    """All metrics of one target, computed over a fixed label list."""

    labels: list[str]
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "labels": list(self.labels),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": {
                label: {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "support": metrics.support,
                }
                for label, metrics in self.per_class.items()
            },
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetMetrics":
        """Rebuild metrics from the output of ``to_dict``."""
        return cls(
            labels=[str(label) for label in data["labels"]],
            accuracy=float(data["accuracy"]),
            macro_f1=float(data["macro_f1"]),
            per_class={
                str(label): ClassMetrics(
                    precision=float(values["precision"]),
                    recall=float(values["recall"]),
                    f1=float(values["f1"]),
                    support=int(values["support"]),
                )
                for label, values in data["per_class"].items()
            },
            confusion_matrix=[[int(cell) for cell in row] for row in data["confusion_matrix"]],
        )


def compute_target_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> TargetMetrics:
    """Compute accuracy, per-class scores, macro F1 and the confusion matrix.

    ``labels`` fixes the order of ``per_class`` and of the confusion matrix
    rows (true) and columns (predicted). Undefined scores become 0.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot compute metrics on an empty sample")
    label_list = list(labels)
    true_list = list(y_true)
    pred_list = list(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_list, pred_list, labels=label_list, zero_division=0
    )
    per_class = {
        label: ClassMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(label_list)
    }
    matrix = confusion_matrix(true_list, pred_list, labels=label_list)
    return TargetMetrics(
        labels=label_list,
        accuracy=float(accuracy_score(true_list, pred_list)),
        macro_f1=float(sum(float(value) for value in f1) / len(label_list)),
        per_class=per_class,
        confusion_matrix=[[int(cell) for cell in row] for row in matrix],
    )


def evaluate_classifier(
    classifier: TicketClassifier, frame: pd.DataFrame
) -> dict[str, TargetMetrics]:
    """Predict ``frame`` in batches and compute metrics for each target.

    ``frame`` must have the columns ``title``, ``description``, ``category``
    and ``priority``.
    """
    titles = [str(value) for value in frame["title"]]
    descriptions = [str(value) for value in frame["description"]]
    predicted_categories: list[str] = []
    predicted_priorities: list[str] = []
    for start in range(0, len(titles), BATCH_SIZE):
        batch = classifier.predict_proba(
            titles[start : start + BATCH_SIZE], descriptions[start : start + BATCH_SIZE]
        )
        for probs in batch:
            predicted_categories.append(top_label(probs.category, CATEGORIES)[0])
            predicted_priorities.append(top_label(probs.priority, PRIORITIES)[0])
    return {
        "category": compute_target_metrics(
            [str(value) for value in frame["category"]], predicted_categories, CATEGORIES
        ),
        "priority": compute_target_metrics(
            [str(value) for value in frame["priority"]], predicted_priorities, PRIORITIES
        ),
    }
