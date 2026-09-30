import logging
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pytest
from fastapi.testclient import TestClient
from support.api_fixtures import build_promoted_baseline

from ticket_classifier.api.app import AppState, create_app
from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.base import ClassProbabilities
from ticket_classifier.services import prediction_service
from ticket_classifier.settings import Settings

API01_KEYS = {
    "prediction_id",
    "category",
    "category_confidence",
    "priority",
    "priority_confidence",
    "top_categories",
    "needs_review",
    "model_version",
}
MODEL_UNAVAILABLE = {
    "code": "MODEL_UNAVAILABLE",
    "message": "Model not available. Try again later.",
}
STORAGE_UNAVAILABLE = {
    "code": "STORAGE_UNAVAILABLE",
    "message": "Service temporarily unavailable. Please try again later.",
}
VALID_TICKET = {"title": "Cannot log in", "description": "My password reset link does not work."}


class FakeClassifier:
    kind: Literal["baseline", "transformer"] = "baseline"

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        category = {"access": 0.7, "infrastructure": 0.1, "billing": 0.1, "bug": 0.05}
        category["other"] = 0.05
        priority = {"low": 0.1, "medium": 0.2, "high": 0.7}
        return [ClassProbabilities(category=category, priority=priority) for _ in titles]

    def save(self, directory: Path) -> None:
        raise NotImplementedError


@pytest.fixture(scope="module")
def promoted_settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    base = tmp_path_factory.mktemp("promoted")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{base}/mlflow.db")
        settings = build_promoted_baseline(base)
    yield settings


@pytest.fixture
def settings(promoted_settings: Settings, tmp_path: Path) -> Settings:
    # Fresh database per test; the promoted model artifacts are shared.
    return promoted_settings.model_copy(update={"db_path": tmp_path / "db" / "tickets.db"})


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _rows(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM predictions")]
    finally:
        conn.close()


def _state(client: TestClient) -> AppState:
    state = client.app.state.ctx  # type: ignore[attr-defined]
    assert isinstance(state, AppState)
    return state


def test_api01_api02_api08_valid_ticket_returns_full_prediction_without_api_key(
    client: TestClient, settings: Settings
) -> None:
    response = client.post("/predict", json=VALID_TICKET)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == API01_KEYS
    assert str(uuid.UUID(body["prediction_id"], version=4)) == body["prediction_id"]
    assert body["category"] in CATEGORIES
    assert body["priority"] in PRIORITIES
    assert 0.0 <= body["category_confidence"] <= 1.0
    assert 0.0 <= body["priority_confidence"] <= 1.0
    assert isinstance(body["needs_review"], bool)
    assert body["model_version"] == _state(client).model_version
    top = body["top_categories"]
    assert len(top) == 3
    assert len({item["category"] for item in top}) == 3
    assert all(set(item) == {"category", "probability"} for item in top)
    probabilities = [item["probability"] for item in top]
    assert probabilities == sorted(probabilities, reverse=True)
    assert top[0] == {"category": body["category"], "probability": body["category_confidence"]}
    rows = _rows(settings.db_path)
    assert [row["prediction_id"] for row in rows] == [body["prediction_id"]]
    assert rows[0]["model_version"] == body["model_version"]


@pytest.mark.parametrize(("threshold", "expected"), [(0.0, False), (1.0, True)])
def test_api03_api04_threshold_from_settings_drives_needs_review(
    settings: Settings, threshold: float, expected: bool
) -> None:
    configured = settings.model_copy(update={"review_threshold": threshold})

    with TestClient(create_app(configured)) as client:
        _state(client).classifier = FakeClassifier()
        response = client.post("/predict", json=VALID_TICKET)

    assert response.status_code == 200
    body = response.json()
    assert body["needs_review"] is expected
    assert body["category"] == "access"
    assert body["priority"] == "high"
    assert _rows(configured.db_path)[0]["needs_review"] == int(expected)


def test_api05_api07_stored_prediction_has_masked_pii_only(
    client: TestClient, settings: Settings
) -> None:
    ticket = {
        "title": "Contact john@acme.com",
        "description": "Please call +1 555 123 4567 about my invoice.",
    }

    response = client.post("/predict", json=ticket)

    assert response.status_code == 200
    rows = _rows(settings.db_path)
    assert len(rows) == 1
    row = rows[0]
    assert "[EMAIL]" in row["title_masked"]
    assert "[PHONE]" in row["description_masked"]
    for value in row.values():
        assert "john@acme.com" not in str(value)
        assert "555 123 4567" not in str(value)


def test_api08_predict_ignores_api_key_header_entirely(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_TICKET, headers={"X-API-Key": "wrong"})

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"description": "text"}, "Field 'title' is required."),
        ({"title": "text"}, "Field 'description' is required."),
        (
            {"title": "a" * 201, "description": "text"},
            "Field 'title' must be between 1 and 200 characters.",
        ),
        (
            {"title": "text", "description": "a" * 5001},
            "Field 'description' must be between 1 and 5000 characters.",
        ),
        ({"title": "   ", "description": "text"}, "Field 'title' must not be blank."),
        ({"title": "text", "description": " \n\t "}, "Field 'description' must not be blank."),
        ({"title": 123, "description": "text"}, "Invalid request body."),
        ({"title": "text", "description": None}, "Invalid request body."),
        ({"title": ["text"], "description": "text"}, "Invalid request body."),
        (["title", "description"], "Invalid request body."),
    ],
)
def test_api90_api91_api92_api93_invalid_body_returns_422_without_storing(
    client: TestClient, settings: Settings, payload: Any, message: str
) -> None:
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json() == {"code": "VALIDATION_ERROR", "message": message}
    assert _rows(settings.db_path) == []


def test_api93_non_json_body_returns_422_without_storing(
    client: TestClient, settings: Settings
) -> None:
    response = client.post(
        "/predict", content=b"title=x", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json() == {"code": "VALIDATION_ERROR", "message": "Invalid request body."}
    assert _rows(settings.db_path) == []


def test_api94_without_promoted_version_returns_503_without_storing(tmp_path: Path) -> None:
    settings = Settings(artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "tickets.db")

    with TestClient(create_app(settings)) as client:
        response = client.post("/predict", json=VALID_TICKET)

    assert response.status_code == 503
    assert response.json() == MODEL_UNAVAILABLE
    assert _rows(settings.db_path) == []


def test_api96_storage_failure_returns_503_without_prediction(
    client: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(*_args: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(prediction_service, "insert_prediction", boom)

    with caplog.at_level(logging.INFO):
        response = client.post("/predict", json=VALID_TICKET)

    assert response.status_code == 503
    assert response.json() == STORAGE_UNAVAILABLE
    assert "prediction_id" not in response.text
    assert _rows(settings.db_path) == []
    assert any("OperationalError" in r.getMessage() for r in caplog.records)


def test_api95_concurrent_requests_get_distinct_ids_and_own_rows(
    client: TestClient, settings: Settings
) -> None:
    words = [chr(ord("a") + i) * 4 for i in range(20)]
    tickets = [
        {"title": f"Ticket {word}", "description": f"Problem with {word} module"} for word in words
    ]

    with ThreadPoolExecutor(max_workers=20) as pool:
        responses = list(pool.map(lambda t: client.post("/predict", json=t), tickets))

    assert all(r.status_code == 200 for r in responses)
    ids = [r.json()["prediction_id"] for r in responses]
    assert len(set(ids)) == 20
    rows = {row["prediction_id"]: row for row in _rows(settings.db_path)}
    assert len(rows) == 20
    for prediction_id, ticket in zip(ids, tickets, strict=True):
        assert rows[prediction_id]["title_masked"] == ticket["title"]
        assert rows[prediction_id]["description_masked"] == ticket["description"]


def test_api06_api91_description_with_5000_characters_is_accepted(client: TestClient) -> None:
    description = ("reset my password " * 300)[:4999] + "z"
    assert len(description) == 5000

    response = client.post("/predict", json={"title": "Long ticket", "description": description})

    assert response.status_code == 200


def test_api07_logs_prediction_metadata_without_ticket_text(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    ticket = {"title": "Secret title words", "description": "Confidential description body"}

    with caplog.at_level(logging.INFO):
        response = client.post("/predict", json=ticket)

    assert response.status_code == 200
    messages = [r.getMessage() for r in caplog.records]
    assert any(response.json()["prediction_id"] in m for m in messages)
    assert all("Secret title" not in m and "Confidential" not in m for m in messages)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"title": "\u0000\u0001", "description": "d"}, "Field 'title' must not be blank."),
        (
            {"title": "Valid title", "description": "\u0000"},
            "Field 'description' must not be blank.",
        ),
    ],
)
def test_api92_lac36_blank_after_preprocessing_returns_422_without_storing(
    client: TestClient, settings: Settings, payload: dict[str, str], message: str
) -> None:
    calls: list[object] = []
    fake = FakeClassifier()
    original = fake.predict_proba

    def tracking(titles: Sequence[str], descriptions: Sequence[str]) -> list[ClassProbabilities]:
        calls.append((titles, descriptions))
        return original(titles, descriptions)

    fake.predict_proba = tracking  # type: ignore[method-assign]
    _state(client).classifier = fake

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json() == {"code": "VALIDATION_ERROR", "message": message}
    assert calls == []
    assert _rows(settings.db_path) == []
