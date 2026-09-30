"""Request and response models for the REST API (plan section 8.1)."""

from typing import Literal

from pydantic import BaseModel, StrictStr, ValidationInfo, field_validator
from pydantic_core import PydanticCustomError

from ticket_classifier.labels import Category, Priority

TITLE_MAX_LENGTH = 200
DESCRIPTION_MAX_LENGTH = 5000

_MAX_LENGTHS: dict[str, int] = {
    "title": TITLE_MAX_LENGTH,
    "description": DESCRIPTION_MAX_LENGTH,
}


class PredictRequest(BaseModel):
    """Ticket text to classify. Both fields are stripped before length checks."""

    title: StrictStr
    description: StrictStr

    @field_validator("title", "description")
    @classmethod
    def _strip_and_check_length(cls, value: str, info: ValidationInfo) -> str:
        stripped = value.strip()
        if not stripped:
            raise PydanticCustomError("blank", "Value must not be blank.")
        max_length = _MAX_LENGTHS[info.field_name or ""]
        if len(stripped) > max_length:
            raise PydanticCustomError(
                "too_long",
                "Value must have at most {max} characters.",
                {"max": max_length},
            )
        return stripped


class TopCategory(BaseModel):
    category: str
    probability: float


class PredictResponse(BaseModel):
    prediction_id: str
    category: Category
    category_confidence: float
    priority: Priority
    priority_confidence: float
    top_categories: list[TopCategory]
    needs_review: bool
    model_version: str


class FeedbackRequest(BaseModel):
    prediction_id: StrictStr
    category: Category
    priority: Priority


class FeedbackResponse(BaseModel):
    feedback_id: str
    prediction_id: str
    category: Category
    priority: Priority
    received_at: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_version: str | None


class ErrorBody(BaseModel):
    code: str
    message: str
