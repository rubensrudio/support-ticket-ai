"""Retention purge: delete old predictions that never received feedback.

Predictions with at least one feedback row are kept forever, and feedback rows
are never deleted. The ``ON DELETE RESTRICT`` foreign key plus a single
``BEGIN IMMEDIATE`` transaction guarantee that no feedback is ever left without
its prediction, even when feedback arrives while the purge is running.
"""

import sqlite3
from datetime import datetime, timedelta

from ticket_classifier.storage.database import to_iso, transaction

PURGE_MESSAGE = (
    "Purge finished: {count} predictions without feedback older than {days} days deleted."
)

_PURGE_SQL = """
DELETE FROM predictions
WHERE created_at < ?
  AND NOT EXISTS (
      SELECT 1 FROM feedback WHERE feedback.prediction_id = predictions.prediction_id
  )
"""


def _close_dangling_transaction(conn: sqlite3.Connection) -> None:
    # ``transaction`` leaves the transaction open when COMMIT itself fails.
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass


def purge_expired_predictions(
    conn: sqlite3.Connection, now: datetime, retention_days: int = 90
) -> int:
    """Delete predictions older than ``retention_days`` without feedback; return the count.

    A prediction is deleted only when ``created_at < now - retention_days``, so one
    stored exactly ``retention_days`` ago is kept. Raises ``ValueError`` if ``now``
    is naive or ``retention_days`` is lower than 1.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")
    cutoff = to_iso(now - timedelta(days=retention_days))
    try:
        with transaction(conn):
            cursor = conn.execute(_PURGE_SQL, (cutoff,))
            deleted = cursor.rowcount
    except BaseException:
        _close_dangling_transaction(conn)
        raise
    return deleted
