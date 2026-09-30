"""SQLite connection, schema and transaction helpers (stdlib ``sqlite3`` only)."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ticket_classifier.labels import CATEGORIES, PRIORITIES


def _in_list(values: tuple[str, ...]) -> str:
    # Label values are fixed module constants, never user input.
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY NOT NULL,
    title_masked TEXT NOT NULL,
    description_masked TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ({_in_list(CATEGORIES)})),
    category_confidence REAL NOT NULL
        CHECK (category_confidence >= 0.0 AND category_confidence <= 1.0),
    priority TEXT NOT NULL CHECK (priority IN ({_in_list(PRIORITIES)})),
    priority_confidence REAL NOT NULL
        CHECK (priority_confidence >= 0.0 AND priority_confidence <= 1.0),
    needs_review INTEGER NOT NULL CHECK (needs_review IN (0, 1)),
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT NOT NULL UNIQUE,
    prediction_id TEXT NOT NULL
        REFERENCES predictions (prediction_id) ON DELETE RESTRICT,
    category TEXT NOT NULL CHECK (category IN ({_in_list(CATEGORIES)})),
    priority TEXT NOT NULL CHECK (priority IN ({_in_list(PRIORITIES)})),
    received_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_prediction
    ON feedback (prediction_id, received_at, seq);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection in autocommit mode with the project PRAGMAs applied."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        isolation_level=None,
        check_same_thread=False,
        timeout=5.0,
    )
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not exist (idempotent)."""
    with transaction(conn):
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                conn.execute(statement)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run the block inside ``BEGIN IMMEDIATE``; commit on success, roll back on error."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def to_iso(dt: datetime) -> str:
    """Format an aware datetime as ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00`` in UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(UTC).isoformat(timespec="microseconds")
