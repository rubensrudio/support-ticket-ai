"""Common error format for the REST API (plan sections 8.1 and 8.3)."""

from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ticket_classifier.labels import CATEGORIES, PRIORITIES

VALIDATION_ERROR = "VALIDATION_ERROR"
INVALID_BODY_MESSAGE = "Invalid request body."

_ALLOWED_VALUES: dict[str, tuple[str, ...]] = {
    "category": CATEGORIES,
    "priority": PRIORITIES,
}


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Build a JSON response in the ``{"code", "message"}`` format."""
    return JSONResponse(status_code=status_code, content={"code": code, "message": message})


def _field_name(error: Mapping[str, Any]) -> str | None:
    """Return the top-level body field an error refers to, or None for body-level errors.

    Accepts FastAPI locations (``("body", "title")``) and plain pydantic locations
    from ``model_validate_json`` (``("category",)``). An empty location or one that
    points at a list index is a body-level error.
    """
    loc = tuple(error.get("loc", ()))
    if loc and loc[0] == "body":
        loc = loc[1:]
    if loc and isinstance(loc[0], str):
        return loc[0]
    return None


def _one_of(field: str) -> str:
    allowed = ", ".join(_ALLOWED_VALUES[field])
    return f"Field '{field}' must be one of: {allowed}."


def _field_message(field: str, error: Mapping[str, Any]) -> str:
    error_type = error.get("type")
    if field in _ALLOWED_VALUES and error_type in ("missing", "literal_error"):
        return _one_of(field)
    if error_type == "missing":
        return f"Field '{field}' is required."
    if error_type == "blank":
        return f"Field '{field}' must not be blank."
    if error_type == "too_long":
        ctx = error.get("ctx") or {}
        max_length = ctx.get("max")
        if isinstance(max_length, int):
            return f"Field '{field}' must be between 1 and {max_length} characters."
    return INVALID_BODY_MESSAGE


def validation_message(errors: Sequence[Mapping[str, Any]]) -> str:
    """Translate request validation errors into a single user-facing message.

    Body-level errors (non-JSON, non-object, missing body) win; otherwise the first
    field error is used, which follows the declared field order of the model.
    """
    field_errors: list[tuple[str, Mapping[str, Any]]] = []
    for error in errors:
        field = _field_name(error)
        if field is None:
            return INVALID_BODY_MESSAGE
        field_errors.append((field, error))
    if not field_errors:
        return INVALID_BODY_MESSAGE
    field, first = field_errors[0]
    return _field_message(field, first)


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    return error_response(422, VALIDATION_ERROR, validation_message(errors))


def register_error_handlers(app: FastAPI) -> None:
    """Replace FastAPI's default 422 body with the common error format."""
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
