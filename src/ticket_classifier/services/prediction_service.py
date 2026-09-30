"""Ticket classification use case behind ``POST /predict`` (CT-24, plan section 5.2).

The ticket text is preprocessed (normalized and PII-masked) before inference and
only the masked text is stored (LAC-09). Ticket text is never logged.
"""

import sqlite3
from typing import cast
from uuid import uuid4

from ticket_classifier.api.app import AppState
from ticket_classifier.api.schemas import PredictResponse, TopCategory
from ticket_classifier.labels import CATEGORIES, PRIORITIES, Category, Priority
from ticket_classifier.models.base import top_label
from ticket_classifier.preprocessing import preprocess_text
from ticket_classifier.storage.database import transaction, utc_now
from ticket_classifier.storage.predictions import PredictionRecord, insert_prediction

TOP_CATEGORIES_SIZE = 3


class ModelUnavailableError(Exception):
    """No promoted model version is loaded (API-94)."""


class PredictionStorageError(Exception):
    """The prediction could not be stored (API-96)."""


def _top_categories(probs: dict[str, float]) -> list[TopCategory]:
    # Stable sort over the CT-1 order: equal probabilities keep that order, which
    # matches the tie-break of ``top_label``.
    ranked = sorted((label for label in CATEGORIES if label in probs), key=lambda c: -probs[c])
    return [
        TopCategory(category=label, probability=float(probs[label]))
        for label in ranked[:TOP_CATEGORIES_SIZE]
    ]


def predict_ticket(
    state: AppState, conn: sqlite3.Connection, title: str, description: str
) -> PredictResponse:
    """Classify one ticket, store the masked prediction and return the API response.

    Raises ``ModelUnavailableError`` when no model is loaded and
    ``PredictionStorageError`` when the prediction cannot be stored; in both cases
    nothing is returned to the caller.
    """
    classifier = state.classifier
    model_version = state.model_version
    if classifier is None or model_version is None:
        raise ModelUnavailableError("No promoted model version is loaded.")

    title_masked = preprocess_text(title)
    description_masked = preprocess_text(description)
    probs = classifier.predict_proba([title_masked], [description_masked])[0]

    category, category_confidence = top_label(probs.category, CATEGORIES)
    priority, priority_confidence = top_label(probs.priority, PRIORITIES)
    category_confidence = float(category_confidence)
    priority_confidence = float(priority_confidence)
    threshold = state.settings.review_threshold
    needs_review = category_confidence < threshold or priority_confidence < threshold

    record = PredictionRecord(
        prediction_id=str(uuid4()),
        title_masked=title_masked,
        description_masked=description_masked,
        category=category,
        category_confidence=category_confidence,
        priority=priority,
        priority_confidence=priority_confidence,
        needs_review=needs_review,
        model_version=model_version,
        created_at=utc_now(),
    )
    try:
        with transaction(conn):
            insert_prediction(conn, record)
    except sqlite3.Error as exc:
        raise PredictionStorageError("Prediction could not be stored.") from exc

    return PredictResponse(
        prediction_id=record.prediction_id,
        category=cast(Category, category),
        category_confidence=category_confidence,
        priority=cast(Priority, priority),
        priority_confidence=priority_confidence,
        top_categories=_top_categories(probs.category),
        needs_review=needs_review,
        model_version=model_version,
    )
