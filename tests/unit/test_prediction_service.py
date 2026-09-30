import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import pytest

from ticket_classifier.api.app import AppState
from ticket_classifier.labels import CATEGORIES
from ticket_classifier.models.base import ClassProbabilities
from ticket_classifier.services import prediction_service
from ticket_classifier.services.prediction_service import (
    ModelUnavailableError,
    PredictionStorageError,
    predict_ticket,
)
from ticket_classifier.settings import Settings
from ticket_classifier.storage.database import connect, init_schema

MODEL_VERSION = "baseline-20260901T000000Z-abcdef12"

DEFAULT_CATEGORY = {
    "access": 0.1,
    "infrastructure": 0.2,
    "billing": 0.05,
    "bug": 0.55,
    "other": 0.1,
}
DEFAULT_PRIORITY = {"low": 0.2, "medium": 0.7, "high": 0.1}


class FakeClassifier:
    kind: Literal["baseline", "transformer"] = "baseline"

    def __init__(
        self,
        category: dict[str, float] | None = None,
        priority: dict[str, float] | None = None,
    ) -> None:
        self.category = category if category is not None else DEFAULT_CATEGORY
        self.priority = priority if priority is not None else DEFAULT_PRIORITY
        self.calls: list[tuple[list[str], list[str]]] = []

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        self.calls.append((list(titles), list(descriptions)))
        return [
            ClassProbabilities(category=dict(self.category), priority=dict(self.priority))
            for _ in titles
        ]

    def save(self, directory: Path) -> None:
        raise NotImplementedError


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(tmp_path / "tickets.db")
    init_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def _state(
    tmp_path: Path,
    classifier: FakeClassifier | None = None,
    threshold: float = 0.6,
    model_version: str | None = MODEL_VERSION,
) -> AppState:
    settings = Settings(
        review_threshold=threshold,
        artifacts_dir=tmp_path / "artifacts",
        db_path=tmp_path / "tickets.db",
    )
    return AppState(settings=settings, classifier=classifier, model_version=model_version)


def _rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM predictions"))
    finally:
        conn.row_factory = None


def test_api01_api02_returns_top1_top3_and_model_version(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    state = _state(tmp_path, FakeClassifier())

    result = predict_ticket(state, conn, "App crashes", "It crashes on save")

    assert str(uuid.UUID(result.prediction_id, version=4)) == result.prediction_id
    assert result.category == "bug"
    assert result.category_confidence == 0.55
    assert result.priority == "medium"
    assert result.priority_confidence == 0.7
    assert result.model_version == MODEL_VERSION
    assert [(t.category, t.probability) for t in result.top_categories] == [
        ("bug", 0.55),
        ("infrastructure", 0.2),
        ("access", 0.1),
    ]


def test_api02_top3_ties_follow_ct1_order_and_first_matches_top1(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    tied = dict.fromkeys(CATEGORIES, 0.2)
    state = _state(tmp_path, FakeClassifier(category=tied))

    result = predict_ticket(state, conn, "t", "d")

    assert result.category == "access"
    assert [t.category for t in result.top_categories] == ["access", "infrastructure", "billing"]
    first = result.top_categories[0]
    assert (first.category, first.probability) == (result.category, result.category_confidence)


@pytest.mark.parametrize(
    ("threshold", "category_conf", "priority_conf", "expected"),
    [
        (0.6, 0.55, 0.7, True),
        (0.6, 0.7, 0.55, True),
        (0.6, 0.6, 0.6, False),
        (0.6, 0.9, 0.8, False),
        (0.0, 0.3, 0.4, False),
        (1.0, 0.99, 0.99, True),
    ],
)
def test_api03_api04_needs_review_uses_configured_threshold(
    tmp_path: Path,
    conn: sqlite3.Connection,
    threshold: float,
    category_conf: float,
    priority_conf: float,
    expected: bool,
) -> None:
    rest = (1.0 - category_conf) / 4
    category = {label: rest for label in CATEGORIES}
    category["billing"] = category_conf
    priority = {"low": (1.0 - priority_conf) / 2, "medium": (1.0 - priority_conf) / 2}
    priority["high"] = priority_conf
    state = _state(tmp_path, FakeClassifier(category, priority), threshold=threshold)

    result = predict_ticket(state, conn, "t", "d")

    assert result.needs_review is expected
    assert result.category == "billing"
    assert result.priority == "high"


def test_api05_api07_preprocesses_input_and_stores_masked_text_only(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    classifier = FakeClassifier()
    state = _state(tmp_path, classifier)

    result = predict_ticket(state, conn, "Mail john@acme.com", "Call me at +1 555 123 4567 please")

    assert classifier.calls == [(["Mail [EMAIL]"], ["Call me at [PHONE] please"])]
    rows = _rows(conn)
    assert len(rows) == 1
    row = rows[0]
    assert row["prediction_id"] == result.prediction_id
    assert row["title_masked"] == "Mail [EMAIL]"
    assert row["description_masked"] == "Call me at [PHONE] please"
    assert row["category"] == "bug"
    assert row["category_confidence"] == 0.55
    assert row["priority"] == "medium"
    assert row["priority_confidence"] == 0.7
    assert row["needs_review"] == 1
    assert row["model_version"] == MODEL_VERSION
    assert row["created_at"].endswith("+00:00")
    for value in tuple(row):
        assert "john@acme.com" not in str(value)
        assert "555 123 4567" not in str(value)


def test_api94_raises_model_unavailable_without_classifier(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    state = _state(tmp_path, None, model_version=None)

    with pytest.raises(ModelUnavailableError):
        predict_ticket(state, conn, "t", "d")

    assert _rows(conn) == []


def test_api94_raises_model_unavailable_without_model_version(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    state = _state(tmp_path, FakeClassifier(), model_version=None)

    with pytest.raises(ModelUnavailableError):
        predict_ticket(state, conn, "t", "d")

    assert _rows(conn) == []


def test_api96_storage_error_becomes_prediction_storage_error(
    tmp_path: Path, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(prediction_service, "insert_prediction", boom)
    state = _state(tmp_path, FakeClassifier())

    with pytest.raises(PredictionStorageError) as excinfo:
        predict_ticket(state, conn, "t", "d")

    assert isinstance(excinfo.value.__cause__, sqlite3.OperationalError)
    assert _rows(conn) == []
    assert not conn.in_transaction


def test_api96_closed_connection_becomes_prediction_storage_error(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    state = _state(tmp_path, FakeClassifier())
    conn.close()

    with pytest.raises(PredictionStorageError):
        predict_ticket(state, conn, "t", "d")
