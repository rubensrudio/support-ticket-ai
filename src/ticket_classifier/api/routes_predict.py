"""``POST /predict``: classify a support ticket (API-01..API-08, API-90..API-96).

Public by decision (AS-6, LAC-08): no API key is required. The handler is
synchronous (FastAPI runs it in a worker thread) and never logs ticket text.
"""

import logging
import sqlite3
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ticket_classifier.api.app import AppState, get_app_state, get_connection
from ticket_classifier.api.errors import VALIDATION_ERROR, error_response
from ticket_classifier.api.schemas import ErrorBody, PredictRequest, PredictResponse
from ticket_classifier.services.prediction_service import (
    BlankFieldError,
    ModelUnavailableError,
    PredictionStorageError,
    predict_ticket,
)

logger = logging.getLogger(__name__)

MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
MODEL_UNAVAILABLE_MESSAGE = "Model not available. Try again later."
STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
STORAGE_UNAVAILABLE_MESSAGE = "Service temporarily unavailable. Please try again later."

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={422: {"model": ErrorBody}, 503: {"model": ErrorBody}},
)
def predict(
    request: PredictRequest,
    state: Annotated[AppState, Depends(get_app_state)],
    conn: Annotated[sqlite3.Connection, Depends(get_connection)],
) -> PredictResponse | JSONResponse:
    started = time.perf_counter()
    try:
        result = predict_ticket(state, conn, request.title, request.description)
    except BlankFieldError as exc:
        return error_response(422, VALIDATION_ERROR, f"Field '{exc.field}' must not be blank.")
    except ModelUnavailableError as exc:
        logger.error("POST /predict returned 503 (%s).", type(exc).__name__)
        return error_response(503, MODEL_UNAVAILABLE, MODEL_UNAVAILABLE_MESSAGE)
    except PredictionStorageError as exc:
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        logger.error("POST /predict returned 503 (%s).", type(cause).__name__)
        return error_response(503, STORAGE_UNAVAILABLE, STORAGE_UNAVAILABLE_MESSAGE)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Prediction %s served by model %s (needs_review=%s, latency=%.1f ms).",
        result.prediction_id,
        result.model_version,
        result.needs_review,
        elapsed_ms,
    )
    return result
