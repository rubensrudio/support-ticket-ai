import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest

from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.base import ClassProbabilities, TicketClassifier, top_label


def test_api02_top_label_returns_argmax_with_exact_value() -> None:
    assert top_label({"access": 0.2, "bug": 0.5, "other": 0.3}, CATEGORIES) == ("bug", 0.5)


def test_api02_top_label_keeps_unrounded_probability() -> None:
    probs = {"low": 0.123456789, "medium": 0.543210987, "high": 0.333332224}

    assert top_label(probs, PRIORITIES) == ("medium", 0.543210987)


def test_api02_top_label_breaks_tie_by_order() -> None:
    probs = {"billing": 0.4, "access": 0.4, "other": 0.2}

    assert top_label(probs, CATEGORIES) == ("access", 0.4)


def test_api02_top_label_tie_follows_given_order() -> None:
    probs = {"billing": 0.4, "access": 0.4, "other": 0.2}

    assert top_label(probs, ("other", "billing", "access")) == ("billing", 0.4)


def test_api02_top_label_rejects_empty_probabilities() -> None:
    with pytest.raises(ValueError):
        top_label({}, CATEGORIES)


def test_api02_top_label_rejects_label_outside_order() -> None:
    with pytest.raises(ValueError):
        top_label({"access": 0.4, "unknown": 0.6}, CATEGORIES)


def test_api02_class_probabilities_is_frozen() -> None:
    probs = ClassProbabilities(
        category={"access": 1.0},
        priority={"low": 1.0},
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        probs.category = {}  # type: ignore[misc]


class _FakeClassifier:
    kind: Literal["baseline", "transformer"] = "baseline"

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        return [ClassProbabilities(category={"other": 1.0}, priority={"low": 1.0}) for _ in titles]

    def save(self, directory: Path) -> None:
        return None


class _Incomplete:
    kind = "baseline"

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        return []


def test_api02_class_with_protocol_members_is_ticket_classifier() -> None:
    assert isinstance(_FakeClassifier(), TicketClassifier)


def test_api02_class_without_save_is_not_ticket_classifier() -> None:
    assert not isinstance(_Incomplete(), TicketClassifier)
