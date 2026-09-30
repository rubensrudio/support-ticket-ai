"""Append-only repository for feedback on stored predictions.

Feedback rows are never updated or deleted. The current feedback of a
prediction is the one with the greatest ``(received_at, seq)``.
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from ticket_classifier.storage.database import to_iso, transaction, utc_now


class PredictionNotFoundError(Exception):
    """Raised when feedback references a prediction that is not stored."""


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    prediction_id: str
    category: str
    priority: str
    received_at: datetime


@dataclass(frozen=True)
class FeedbackTrainingRow:
    prediction_id: str
    title: str
    description: str
    category: str
    priority: str


_CURRENT_FEEDBACK_SQL = """
SELECT p.prediction_id, p.title_masked, p.description_masked, f.category, f.priority
FROM (
    SELECT prediction_id, category, priority,
           ROW_NUMBER() OVER (
               PARTITION BY prediction_id ORDER BY received_at DESC, seq DESC
           ) AS rank
    FROM feedback
) AS f
JOIN predictions AS p ON p.prediction_id = f.prediction_id
WHERE f.rank = 1
ORDER BY p.prediction_id
"""


def _is_foreign_key_error(exc: sqlite3.IntegrityError) -> bool:
    return "FOREIGN KEY" in str(exc).upper()


def _close_dangling_transaction(conn: sqlite3.Connection) -> None:
    # ``transaction`` leaves the transaction open when COMMIT itself fails.
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass


def insert_feedback(
    conn: sqlite3.Connection,
    prediction_id: str,
    category: str,
    priority: str,
    received_at: datetime | None = None,
) -> FeedbackRecord:
    """Append one feedback row for an existing prediction.

    Raises ``PredictionNotFoundError`` if the prediction is not stored,
    ``ValueError`` if ``received_at`` is naive and ``sqlite3.IntegrityError``
    on any other constraint violation (e.g. unknown label).
    """
    moment = utc_now() if received_at is None else received_at
    received_iso = to_iso(moment)
    record = FeedbackRecord(
        feedback_id=str(uuid.uuid4()),
        prediction_id=prediction_id,
        category=category,
        priority=priority,
        received_at=moment.astimezone(UTC),
    )
    try:
        with transaction(conn):
            exists = conn.execute(
                "SELECT 1 FROM predictions WHERE prediction_id = ? LIMIT 1", (prediction_id,)
            ).fetchone()
            if exists is None:
                raise PredictionNotFoundError(prediction_id)
            conn.execute(
                "INSERT INTO feedback (feedback_id, prediction_id, category, priority, received_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (record.feedback_id, prediction_id, category, priority, received_iso),
            )
    except sqlite3.IntegrityError as exc:
        _close_dangling_transaction(conn)
        if _is_foreign_key_error(exc):
            raise PredictionNotFoundError(prediction_id) from None
        raise
    except BaseException:
        _close_dangling_transaction(conn)
        raise
    return record


def list_current_feedback(conn: sqlite3.Connection) -> list[FeedbackTrainingRow]:
    """Return the current feedback per prediction with its masked texts, by ``prediction_id``."""
    return [
        FeedbackTrainingRow(
            prediction_id=row[0],
            title=row[1],
            description=row[2],
            category=row[3],
            priority=row[4],
        )
        for row in conn.execute(_CURRENT_FEEDBACK_SQL).fetchall()
    ]
