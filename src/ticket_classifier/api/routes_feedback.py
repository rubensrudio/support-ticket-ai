"""``POST /feedback``: record the correct labels for a stored prediction.

DA-13: the body is read manually so the API key is checked before any JSON
parsing (a non-JSON body without a key must get 401, not 422). The request
schema is published through ``openapi_extra``. The API key is never logged.
"""

import logging
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from ticket_classifier.api.app import AppState, get_app_state, get_connection
from ticket_classifier.api.errors import VALIDATION_ERROR, error_response, validation_message
from ticket_classifier.api.schemas import ErrorBody, FeedbackRequest, FeedbackResponse
from ticket_classifier.api.security import API_KEY_HEADER, is_authorized
from ticket_classifier.storage.database import to_iso
from ticket_classifier.storage.feedback import PredictionNotFoundError, insert_feedback

logger = logging.getLogger(__name__)

UNAUTHORIZED = "UNAUTHORIZED"
UNAUTHORIZED_MESSAGE = "Invalid or missing API key."
PREDICTION_NOT_FOUND = "PREDICTION_NOT_FOUND"
STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
STORAGE_UNAVAILABLE_MESSAGE = "Service temporarily unavailable. Please try again later."

router = APIRouter()

_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {"application/json": {"schema": FeedbackRequest.model_json_schema()}},
    },
    "parameters": [
        {
            "name": API_KEY_HEADER,
            "in": "header",
            "required": True,
            "schema": {"type": "string"},
            "description": "API key authorizing feedback submission.",
        }
    ],
}


@router.post(
    "/feedback",
    status_code=201,
    response_model=FeedbackResponse,
    responses={
        401: {"model": ErrorBody},
        404: {"model": ErrorBody},
        422: {"model": ErrorBody},
        503: {"model": ErrorBody},
    },
    openapi_extra=_OPENAPI_EXTRA,
)
async def create_feedback(
    request: Request,
    state: Annotated[AppState, Depends(get_app_state)],
    conn: Annotated[sqlite3.Connection, Depends(get_connection)],
) -> FeedbackResponse | JSONResponse:
    if not is_authorized(state.settings, request.headers.get(API_KEY_HEADER)):
        logger.warning("POST /feedback returned 401 (invalid or missing API key).")
        return error_response(401, UNAUTHORIZED, UNAUTHORIZED_MESSAGE)

    try:
        payload = FeedbackRequest.model_validate_json(await request.body())
    except ValidationError as exc:
        return error_response(422, VALIDATION_ERROR, validation_message(exc.errors()))

    try:
        record = await run_in_threadpool(
            insert_feedback, conn, payload.prediction_id, payload.category, payload.priority
        )
    except PredictionNotFoundError:
        return error_response(
            404, PREDICTION_NOT_FOUND, f"Prediction '{payload.prediction_id}' not found."
        )
    except sqlite3.Error as exc:
        logger.error("POST /feedback returned 503 (%s).", type(exc).__name__)
        return error_response(503, STORAGE_UNAVAILABLE, STORAGE_UNAVAILABLE_MESSAGE)

    logger.info("Feedback %s stored for prediction %s.", record.feedback_id, record.prediction_id)
    return FeedbackResponse(
        feedback_id=record.feedback_id,
        prediction_id=record.prediction_id,
        category=payload.category,
        priority=payload.priority,
        received_at=to_iso(record.received_at),
    )
