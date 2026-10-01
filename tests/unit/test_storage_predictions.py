import re
import sqlite3
import threading
import uuid
from dataclasses import fields
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ticket_classifier.storage.database import (
    connect,
    init_schema,
    to_iso,
    transaction,
    utc_now,
)
from ticket_classifier.storage.predictions import (
    PredictionRecord,
    insert_prediction,
    prediction_exists,
)

ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")


def _record(**overrides: object) -> PredictionRecord:
    values: dict[str, object] = {
        "prediction_id": str(uuid.uuid4()),
        "title_masked": "cannot login [EMAIL]",
        "description_masked": "my account is locked",
        "category": "access",
        "category_confidence": 0.91,
        "priority": "high",
        "priority_confidence": 0.72,
        "needs_review": False,
        "model_version": "baseline-20260901",
        "created_at": datetime(2026, 9, 1, 12, 30, 45, 123456, tzinfo=UTC),
    }
    values.update(overrides)
    return PredictionRecord(**values)  # type: ignore[arg-type]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "dir" / "tickets.db"


@pytest.fixture
def conn(db_path: Path) -> sqlite3.Connection:
    connection = connect(db_path)
    init_schema(connection)
    yield connection  # type: ignore[misc]
    connection.close()


def test_connect_creates_parent_dir_and_sets_pragmas(db_path: Path) -> None:
    connection = connect(db_path)
    try:
        assert db_path.parent.is_dir()
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert connection.isolation_level is None
    finally:
        connection.close()


def test_init_schema_is_idempotent(db_path: Path) -> None:
    connection = connect(db_path)
    try:
        init_schema(connection)
        init_schema(connection)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        indexes = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        assert {"predictions", "feedback"} <= tables
        assert {"idx_predictions_created_at", "idx_feedback_prediction"} <= indexes
    finally:
        connection.close()


def test_feedback_table_enforces_foreign_key(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO feedback (feedback_id, prediction_id, category, priority, received_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "missing", "bug", "low", to_iso(utc_now())),
        )


def test_utc_now_is_aware_utc() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_to_iso_fixed_format_with_microseconds() -> None:
    assert to_iso(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == "2026-01-02T03:04:05.000000+00:00"


def test_to_iso_converts_other_offsets_to_utc() -> None:
    dt = datetime(2026, 1, 2, 5, 4, 5, 7, tzinfo=timezone(timedelta(hours=2)))
    assert to_iso(dt) == "2026-01-02T03:04:05.000007+00:00"


def test_to_iso_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        to_iso(datetime(2026, 1, 2, 3, 4, 5))


def test_api_07_insert_prediction_round_trip(conn: sqlite3.Connection) -> None:
    record = _record()
    insert_prediction(conn, record)

    row = conn.execute(
        "SELECT prediction_id, title_masked, description_masked, category, "
        "category_confidence, priority, priority_confidence, needs_review, "
        "model_version, created_at FROM predictions WHERE prediction_id = ?",
        (record.prediction_id,),
    ).fetchone()

    assert row == (
        record.prediction_id,
        record.title_masked,
        record.description_masked,
        record.category,
        record.category_confidence,
        record.priority,
        record.priority_confidence,
        0,
        record.model_version,
        "2026-09-01T12:30:45.123456+00:00",
    )
    assert ISO_PATTERN.match(row[9])


def test_api_07_needs_review_stored_as_integer(conn: sqlite3.Connection) -> None:
    record = _record(needs_review=True)
    insert_prediction(conn, record)
    stored = conn.execute(
        "SELECT needs_review FROM predictions WHERE prediction_id = ?", (record.prediction_id,)
    ).fetchone()[0]
    assert stored == 1


def test_api_07_record_has_no_original_text_field() -> None:
    names = {f.name for f in fields(PredictionRecord)}
    assert "title" not in names
    assert "description" not in names


def test_insert_invalid_category_raises_integrity_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        insert_prediction(conn, _record(category="hardware"))


def test_insert_invalid_priority_raises_integrity_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        insert_prediction(conn, _record(priority="urgent"))


def test_prediction_exists(conn: sqlite3.Connection) -> None:
    record = _record()
    assert prediction_exists(conn, record.prediction_id) is False
    insert_prediction(conn, record)
    assert prediction_exists(conn, record.prediction_id) is True


def test_transaction_rolls_back_on_error(conn: sqlite3.Connection) -> None:
    record = _record()
    with pytest.raises(RuntimeError):
        with transaction(conn):
            insert_prediction(conn, record)
            raise RuntimeError("boom")
    assert prediction_exists(conn, record.prediction_id) is False


def test_transaction_commits(conn: sqlite3.Connection) -> None:
    record = _record()
    with transaction(conn) as tx:
        insert_prediction(tx, record)
    assert prediction_exists(conn, record.prediction_id) is True


def test_api_95_concurrent_inserts_keep_distinct_ids(db_path: Path) -> None:
    setup = connect(db_path)
    init_schema(setup)
    setup.close()

    count = 20
    barrier = threading.Barrier(count, timeout=30)
    errors: list[BaseException] = []
    expected: dict[str, str] = {}
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            connection = connect(db_path)
            try:
                record = _record(
                    title_masked=f"title {index}",
                    description_masked=f"description {index}",
                    created_at=utc_now(),
                )
                with lock:
                    expected[record.prediction_id] = record.title_masked
                barrier.wait()
                with transaction(connection):
                    insert_prediction(connection, record)
            finally:
                connection.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    check = connect(db_path)
    try:
        rows = check.execute("SELECT prediction_id, title_masked FROM predictions").fetchall()
    finally:
        check.close()
    assert len(rows) == count
    assert len({row[0] for row in rows}) == count
    assert dict(rows) == expected
