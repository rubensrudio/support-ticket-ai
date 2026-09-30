import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ticket_classifier.api.errors import (
    error_response,
    register_error_handlers,
    validation_message,
)
from ticket_classifier.api.schemas import FeedbackRequest, PredictRequest

CATEGORY_ONE_OF = "Field 'category' must be one of: access, infrastructure, billing, bug, other."
PRIORITY_ONE_OF = "Field 'priority' must be one of: low, medium, high."
INVALID_BODY = "Invalid request body."


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    register_error_handlers(app)

    @app.post("/predict")
    def predict(body: PredictRequest) -> dict[str, str]:
        return {"title": body.title, "description": body.description}

    @app.post("/feedback")
    def feedback(body: FeedbackRequest) -> dict[str, str]:
        return {
            "prediction_id": body.prediction_id,
            "category": body.category,
            "priority": body.priority,
        }

    return TestClient(app)


def _assert_validation_error(response: Any, message: str) -> None:
    assert response.status_code == 422
    body = response.json()
    assert body == {"code": "VALIDATION_ERROR", "message": message}
    assert "detail" not in body


def _valid_feedback(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"prediction_id": "p-1", "category": "bug", "priority": "low"}
    payload.update(overrides)
    return payload


def test_api90_missing_title_is_required(client: TestClient) -> None:
    response = client.post("/predict", json={"description": "d"})

    _assert_validation_error(response, "Field 'title' is required.")


def test_api90_missing_description_is_required(client: TestClient) -> None:
    response = client.post("/predict", json={"title": "t"})

    _assert_validation_error(response, "Field 'description' is required.")


def test_api90_first_error_follows_declared_field_order(client: TestClient) -> None:
    response = client.post("/predict", json={"description": "   "})

    _assert_validation_error(response, "Field 'title' is required.")


def test_api91_title_over_200_characters(client: TestClient) -> None:
    response = client.post("/predict", json={"title": "a" * 201, "description": "d"})

    _assert_validation_error(response, "Field 'title' must be between 1 and 200 characters.")


def test_api91_description_over_5000_characters(client: TestClient) -> None:
    response = client.post("/predict", json={"title": "t", "description": "a" * 5001})

    _assert_validation_error(response, "Field 'description' must be between 1 and 5000 characters.")


def test_api91_limit_applies_after_strip(client: TestClient) -> None:
    title = "  " + "a" * 200 + "  "
    response = client.post("/predict", json={"title": title, "description": " d "})

    assert response.status_code == 200
    assert response.json() == {"title": "a" * 200, "description": "d"}


def test_api92_blank_title(client: TestClient) -> None:
    response = client.post("/predict", json={"title": "   ", "description": "d"})

    _assert_validation_error(response, "Field 'title' must not be blank.")


def test_api92_blank_description(client: TestClient) -> None:
    response = client.post("/predict", json={"title": "t", "description": "\t \n"})

    _assert_validation_error(response, "Field 'description' must not be blank.")


def test_api92_empty_title_is_blank(client: TestClient) -> None:
    response = client.post("/predict", json={"title": "", "description": "d"})

    _assert_validation_error(response, "Field 'title' must not be blank.")


def test_api93_non_json_body(client: TestClient) -> None:
    response = client.post(
        "/predict", content=b"not json", headers={"Content-Type": "application/json"}
    )

    _assert_validation_error(response, INVALID_BODY)


def test_api93_non_object_body(client: TestClient) -> None:
    response = client.post("/predict", json=[])

    _assert_validation_error(response, INVALID_BODY)


def test_api93_missing_body(client: TestClient) -> None:
    response = client.post("/predict")

    _assert_validation_error(response, INVALID_BODY)


@pytest.mark.parametrize("value", [5, None, ["a"], {"a": 1}, True])
def test_api93_non_text_title(client: TestClient, value: Any) -> None:
    response = client.post("/predict", json={"title": value, "description": "d"})

    _assert_validation_error(response, INVALID_BODY)


def test_api93_non_text_description(client: TestClient) -> None:
    response = client.post("/predict", json={"title": "t", "description": 1.5})

    _assert_validation_error(response, INVALID_BODY)


def test_api_extra_fields_are_ignored(client: TestClient) -> None:
    response = client.post("/predict", json={"title": "t", "description": "d", "x": 1})

    assert response.status_code == 200


def test_fdbk92_category_outside_set(client: TestClient) -> None:
    response = client.post("/feedback", json=_valid_feedback(category="hardware"))

    _assert_validation_error(response, CATEGORY_ONE_OF)


def test_fdbk92_missing_category(client: TestClient) -> None:
    payload = _valid_feedback()
    del payload["category"]
    response = client.post("/feedback", json=payload)

    _assert_validation_error(response, CATEGORY_ONE_OF)


def test_fdbk92_non_text_category_lists_allowed_values(client: TestClient) -> None:
    response = client.post("/feedback", json=_valid_feedback(category=3))

    _assert_validation_error(response, CATEGORY_ONE_OF)


def test_fdbk92_invalid_priority(client: TestClient) -> None:
    response = client.post("/feedback", json=_valid_feedback(priority="urgent"))

    _assert_validation_error(response, PRIORITY_ONE_OF)


def test_fdbk92_missing_priority(client: TestClient) -> None:
    payload = _valid_feedback()
    del payload["priority"]
    response = client.post("/feedback", json=payload)

    _assert_validation_error(response, PRIORITY_ONE_OF)


def test_fdbk92_category_checked_before_priority(client: TestClient) -> None:
    response = client.post(
        "/feedback", json=_valid_feedback(category="hardware", priority="urgent")
    )

    _assert_validation_error(response, CATEGORY_ONE_OF)


def test_feedback_missing_prediction_id_is_required(client: TestClient) -> None:
    payload = _valid_feedback(category="hardware")
    del payload["prediction_id"]
    response = client.post("/feedback", json=payload)

    _assert_validation_error(response, "Field 'prediction_id' is required.")


def test_feedback_non_text_prediction_id_is_invalid_body(client: TestClient) -> None:
    response = client.post("/feedback", json=_valid_feedback(prediction_id=7))

    _assert_validation_error(response, INVALID_BODY)


def test_feedback_valid_body_is_accepted(client: TestClient) -> None:
    response = client.post("/feedback", json=_valid_feedback())

    assert response.status_code == 200


def test_error_response_builds_code_and_message() -> None:
    response = error_response(404, "PREDICTION_NOT_FOUND", "Prediction 'x' not found.")

    assert response.status_code == 404
    assert json.loads(response.body) == {
        "code": "PREDICTION_NOT_FOUND",
        "message": "Prediction 'x' not found.",
    }


def test_validation_message_body_error_wins_over_field_errors() -> None:
    errors = [
        {"type": "missing", "loc": ("body", "title"), "msg": "Field required"},
        {"type": "json_invalid", "loc": ("body", 3), "msg": "JSON decode error"},
    ]

    assert validation_message(errors) == INVALID_BODY


def test_validation_message_unknown_error_type_is_invalid_body() -> None:
    errors = [{"type": "something_else", "loc": ("body", "title"), "msg": "?"}]

    assert validation_message(errors) == INVALID_BODY


def test_validation_message_empty_errors_is_invalid_body() -> None:
    assert validation_message([]) == INVALID_BODY


def _model_validate_json_errors(raw: str) -> list[Any]:
    with pytest.raises(ValidationError) as exc_info:
        FeedbackRequest.model_validate_json(raw)
    return list(exc_info.value.errors())


def test_fdbk92_model_validate_json_invalid_category_is_one_of() -> None:
    errors = _model_validate_json_errors(json.dumps(_valid_feedback(category="hardware")))

    assert validation_message(errors) == CATEGORY_ONE_OF


def test_fdbk92_model_validate_json_invalid_priority_is_one_of() -> None:
    errors = _model_validate_json_errors(json.dumps(_valid_feedback(priority="urgent")))

    assert validation_message(errors) == PRIORITY_ONE_OF


def test_fdbk92_model_validate_json_missing_prediction_id_is_required() -> None:
    payload = _valid_feedback()
    del payload["prediction_id"]
    errors = _model_validate_json_errors(json.dumps(payload))

    assert validation_message(errors) == "Field 'prediction_id' is required."


@pytest.mark.parametrize("raw", ["not json", "[]"])
def test_fdbk92_model_validate_json_invalid_body(raw: str) -> None:
    errors = _model_validate_json_errors(raw)

    assert validation_message(errors) == INVALID_BODY
