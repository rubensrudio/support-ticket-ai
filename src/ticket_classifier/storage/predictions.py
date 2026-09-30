"""Repository for stored predictions.

The repository persists values as received: masking is the caller's
responsibility, and the record has no field for the original text.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ticket_classifier.storage.database import to_iso


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    title_masked: str
    description_masked: str
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    needs_review: bool
    model_version: str
    created_at: datetime


def insert_prediction(conn: sqlite3.Connection, record: PredictionRecord) -> None:
    """Insert one prediction row. Raises ``sqlite3.IntegrityError`` on constraint violation."""
    conn.execute(
        "INSERT INTO predictions ("
        "prediction_id, title_masked, description_masked, category, category_confidence, "
        "priority, priority_confidence, needs_review, model_version, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.prediction_id,
            record.title_masked,
            record.description_masked,
            record.category,
            record.category_confidence,
            record.priority,
            record.priority_confidence,
            1 if record.needs_review else 0,
            record.model_version,
            to_iso(record.created_at),
        ),
    )


def prediction_exists(conn: sqlite3.Connection, prediction_id: str) -> bool:
    """Return whether a prediction with the given id is stored."""
    row = conn.execute(
        "SELECT 1 FROM predictions WHERE prediction_id = ? LIMIT 1", (prediction_id,)
    ).fetchone()
    return row is not None
