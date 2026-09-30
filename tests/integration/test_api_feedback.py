import logging
import sqlite3
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from support.api_fixtures import TEST_API_KEY, build_promoted_baseline

from ticket_classifier.api import routes_feedback
from ticket_classifier.api.app import create_app
from ticket_classifier.settings import Settings

VALID_TICKET = {"title": "Cannot log in", "description": "My password reset link does not work."}
FEEDBACK_KEYS = {"feedback_id", "prediction_id", "category", "priority", "received_at"}
UNAUTHORIZED = {"code": "UNAUTHORIZED", "message": "Invalid or missing API key."}
STORAGE_UNAVAILABLE = {
    "code": "STORAGE_UNAVAILABLE",
    "message": "Service temporarily unavailable. Please try again later.",
}
AUTH = {"X-API-Key": TEST_API_KEY}


@pytest.fixture(scope="module")
def promoted_settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    base = tmp_path_factory.mktemp("promoted")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{base}/mlflow.db")
        settings = build_promoted_baseline(base)
    yield settings


@pytest.fixture
def settings(promoted_settings: Settings, tmp_path: Path) -> Settings:
    return promoted_settings.model_copy(update={"db_path": tmp_path / "db" / "tickets.db"})


@pytest.fixture
def client(settings: Settings, caplog: pytest.LogCaptureFixture) -> Iterator[TestClient]:
    caplog.set_level(logging.DEBUG)
    with TestClient(create_app(settings)) as test_client:
        yield test_client
    assert TEST_API_KEY not in caplog.text
    for record in caplog.records:
        assert TEST_API_KEY not in record.getMessage()


def _feedback_rows(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM feedback ORDER BY seq")]
    finally:
        conn.close()


def _predict(client: TestClient) -> dict[str, Any]:
    response = client.post("/predict", json=VALID_TICKET)
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def test_fdbk01_valid_key_and_existing_prediction_returns_201_and_stores_row(
    client: TestClient, settings: Settings
) -> None:
    prediction_id = _predict(client)["prediction_id"]

    response = client.post(
        "/feedback",
        json={"prediction_id": prediction_id, "category": "billing", "priority": "low"},
        headers=AUTH,
    )

    assert response.status_code == 201
    body = response.json()
    assert set(body) == FEEDBACK_KEYS
    assert str(uuid.UUID(body["feedback_id"], version=4)) == body["feedback_id"]
    assert body["prediction_id"] == prediction_id
    assert body["category"] == "billing"
    assert body["priority"] == "low"
    received_at = datetime.fromisoformat(body["received_at"])
    assert received_at.utcoffset() is not None
    assert received_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
    rows = _feedback_rows(settings.db_path)
    assert len(rows) == 1
    assert rows[0]["feedback_id"] == body["feedback_id"]
    assert rows[0]["prediction_id"] == prediction_id
    assert (rows[0]["category"], rows[0]["priority"]) == ("billing", "low")


def test_fdbk02_second_feedback_for_same_prediction_keeps_both(
    client: TestClient, settings: Settings
) -> None:
    prediction_id = _predict(client)["prediction_id"]

    first = client.post(
        "/feedback",
        json={"prediction_id": prediction_id, "category": "billing", "priority": "low"},
        headers=AUTH,
    )
    second = client.post(
        "/feedback",
        json={"prediction_id": prediction_id, "category": "bug", "priority": "high"},
        headers=AUTH,
    )

    assert first.status_code == 201
    assert second.status_code == 201
    rows = _feedback_rows(settings.db_path)
    assert [row["feedback_id"] for row in rows] == [
        first.json()["feedback_id"],
        second.json()["feedback_id"],
    ]
    assert [(row["category"], row["priority"]) for row in rows] == [
        ("billing", "low"),
        ("bug", "high"),
    ]


def test_fdbk04_feedback_equal_to_prediction_is_accepted(
    client: TestClient, settings: Settings
) -> None:
    prediction = _predict(client)

    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction["prediction_id"],
            "category": prediction["category"],
            "priority": prediction["priority"],
        },
        headers=AUTH,
    )

    assert response.status_code == 201
    assert len(_feedback_rows(settings.db_path)) == 1


@pytest.mark.parametrize(
    ("headers", "kwargs"),
    [
        ({}, {"json": None}),
        ({"X-API-Key": "wrong-key"}, {"json": None}),
        ({"X-API-Key": ""}, {"json": None}),
        ({"X-API-Key": TEST_API_KEY + " "}, {"json": None}),
        ({}, {"content": b"not json"}),
        ({"X-API-Key": "wrong-key"}, {"content": b"not json"}),
    ],
    ids=["no-header", "wrong-key", "empty-key", "key-with-suffix", "not-json", "not-json-wrong"],
)
def test_fdbk90_missing_or_invalid_key_returns_401_without_storing(
    client: TestClient, settings: Settings, headers: dict[str, str], kwargs: dict[str, Any]
) -> None:
    prediction_id = _predict(client)["prediction_id"]
    if kwargs.get("json", "") is None:
        kwargs = {"json": {"prediction_id": prediction_id, "category": "bug", "priority": "low"}}

    response = client.post("/feedback", headers=headers, **kwargs)

    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED
    assert _feedback_rows(settings.db_path) == []


def test_fdbk93_without_configured_key_rejects_feedback_but_serves_predict_and_health(
    promoted_settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = promoted_settings.model_copy(
        update={"db_path": tmp_path / "db" / "tickets.db", "api_key": None}
    )
    caplog.set_level(logging.DEBUG)
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=VALID_TICKET)
        responses = [
            client.post(
                "/feedback",
                json={
                    "prediction_id": prediction.json()["prediction_id"],
                    "category": "bug",
                    "priority": "low",
                },
                headers=headers,
            )
            for headers in ({}, {"X-API-Key": "anything"}, {"X-API-Key": TEST_API_KEY})
        ]

    assert health.status_code == 200
    assert prediction.status_code == 200
    assert [r.status_code for r in responses] == [401, 401, 401]
    assert all(r.json() == UNAUTHORIZED for r in responses)
    assert _feedback_rows(settings.db_path) == []
    assert TEST_API_KEY not in caplog.text


def test_fdbk91_unknown_prediction_returns_404_without_storing(
    client: TestClient, settings: Settings
) -> None:
    _predict(client)
    missing = str(uuid.uuid4())

    response = client.post(
        "/feedback",
        json={"prediction_id": missing, "category": "bug", "priority": "low"},
        headers=AUTH,
    )

    assert response.status_code == 404
    assert response.json() == {
        "code": "PREDICTION_NOT_FOUND",
        "message": f"Prediction '{missing}' not found.",
    }
    assert _feedback_rows(settings.db_path) == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"category": "hardware", "priority": "low"},
            "Field 'category' must be one of: access, infrastructure, billing, bug, other.",
        ),
        (
            {"priority": "low"},
            "Field 'category' must be one of: access, infrastructure, billing, bug, other.",
        ),
        (
            {"category": "bug", "priority": "urgent"},
            "Field 'priority' must be one of: low, medium, high.",
        ),
        ({"category": "bug"}, "Field 'priority' must be one of: low, medium, high."),
    ],
    ids=["category-invalid", "category-missing", "priority-invalid", "priority-missing"],
)
def test_fdbk92_invalid_or_missing_label_returns_422_with_allowed_values(
    client: TestClient, settings: Settings, payload: dict[str, str], message: str
) -> None:
    prediction_id = _predict(client)["prediction_id"]

    response = client.post(
        "/feedback", json={"prediction_id": prediction_id, **payload}, headers=AUTH
    )

    assert response.status_code == 422
    assert response.json() == {"code": "VALIDATION_ERROR", "message": message}
    assert _feedback_rows(settings.db_path) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"content": b"not json"}, "Invalid request body."),
        ({"content": b""}, "Invalid request body."),
        ({"json": ["bug"]}, "Invalid request body."),
        ({"json": {"category": "bug", "priority": "low"}}, "Field 'prediction_id' is required."),
        (
            {"json": {"prediction_id": 123, "category": "bug", "priority": "low"}},
            "Invalid request body.",
        ),
    ],
    ids=["not-json", "empty", "not-object", "missing-id", "id-not-text"],
)
def test_fdbk92_invalid_body_with_valid_key_returns_422(
    client: TestClient, settings: Settings, kwargs: dict[str, Any], message: str
) -> None:
    response = client.post("/feedback", headers=AUTH, **kwargs)

    assert response.status_code == 422
    assert response.json() == {"code": "VALIDATION_ERROR", "message": message}
    assert _feedback_rows(settings.db_path) == []


def test_fdbk95_storage_failure_returns_503_without_storing(
    client: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prediction_id = _predict(client)["prediction_id"]

    def failing_insert(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(routes_feedback, "insert_feedback", failing_insert)

    response = client.post(
        "/feedback",
        json={"prediction_id": prediction_id, "category": "bug", "priority": "low"},
        headers=AUTH,
    )

    assert response.status_code == 503
    assert response.json() == STORAGE_UNAVAILABLE
    assert _feedback_rows(settings.db_path) == []
    assert any("OperationalError" in r.getMessage() for r in caplog.records)


def test_fdbk90_predict_and_health_do_not_require_key(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json=VALID_TICKET).status_code == 200


def test_fdbk01_openapi_describes_feedback_body(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/feedback"]["post"]
    body_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    if "$ref" in body_schema:
        name = body_schema["$ref"].rsplit("/", 1)[-1]
        body_schema = schema["components"]["schemas"][name]
    assert {"prediction_id", "category", "priority"} <= set(body_schema["properties"])
    assert set(body_schema["required"]) == {"prediction_id", "category", "priority"}
    assert "201" in operation["responses"]
    assert "401" in operation["responses"]
