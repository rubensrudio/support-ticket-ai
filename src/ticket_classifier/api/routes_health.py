"""``GET /health``: public liveness and model status (OPS-01, API-94).

Public by decision (AS-6, LAC-08): no API key is required and the response only
exposes the loaded model version identifier.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ticket_classifier.api.app import AppState, get_app_state
from ticket_classifier.api.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
def health(state: Annotated[AppState, Depends(get_app_state)]) -> JSONResponse:
    if state.classifier is None or state.model_version is None:
        body = HealthResponse(status="unavailable", model_version=None)
        return JSONResponse(status_code=503, content=body.model_dump())
    body = HealthResponse(status="ok", model_version=state.model_version)
    return JSONResponse(status_code=200, content=body.model_dump())
