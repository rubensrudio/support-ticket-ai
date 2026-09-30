import sqlite3
import threading
import uuid
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ticket_classifier.storage.database import connect, init_schema, to_iso
from ticket_classifier.storage.feedback import (
    FeedbackRecord,
    FeedbackTrainingRow,
    PredictionNotFoundError,
    insert_feedback,
    list_current_feedback,
)
from ticket_classifier.storage.predictions import PredictionRecord, insert_prediction

BASE_TIME = datetime(2026, 10, 1, 12, 0, 0, 123456, tzinfo=UTC)


def _prediction(**overrides: object) -> PredictionRecord:
    values: dict[str, object] = {
        "prediction_id": str(uuid.uuid4()),
        "title_masked": "cannot login [EMAIL]",
        "description_masked": "my account is locked [NUMBER]",
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
    return tmp_path / "tickets.db"


@pytest.fixture
def conn(db_path: Path) -> sqlite3.Connection:
    connection = connect(db_path)
    init_schema(connection)
    yield connection  # type: ignore[misc]
    connection.close()


def _store_prediction(conn: sqlite3.Connection, **overrides: object) -> PredictionRecord:
    record = _prediction(**overrides)
    insert_prediction(conn, record)
    return record


def _feedback_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    return conn.execute(
        "SELECT seq, feedback_id, prediction_id, category, priority, received_at "
        "FROM feedback ORDER BY seq"
    ).fetchall()


def test_fdbk_01_insert_feedback_stores_row_and_returns_record(
    conn: sqlite3.Connection,
) -> None:
    prediction = _store_prediction(conn)

    record = insert_feedback(conn, prediction.prediction_id, "billing", "low", BASE_TIME)

    assert isinstance(record, FeedbackRecord)
    assert record.prediction_id == prediction.prediction_id
    assert record.category == "billing"
    assert record.priority == "low"
    assert record.received_at == BASE_TIME
    assert str(uuid.UUID(record.feedback_id, version=4)) == record.feedback_id
    rows = _feedback_rows(conn)
    assert len(rows) == 1
    assert rows[0][1:] == (
        record.feedback_id,
        prediction.prediction_id,
        "billing",
        "low",
        to_iso(BASE_TIME),
    )
    assert not conn.in_transaction


def test_fdbk_01_received_at_defaults_to_now_utc(conn: sqlite3.Connection) -> None:
    prediction = _store_prediction(conn)
    before = datetime.now(UTC)

    record = insert_feedback(conn, prediction.prediction_id, "billing", "low")

    after = datetime.now(UTC)
    assert record.received_at.tzinfo is not None
    assert before <= record.received_at <= after
    assert _feedback_rows(conn)[0][5] == to_iso(record.received_at)


def test_fdbk_01_records_are_frozen(conn: sqlite3.Connection) -> None:
    prediction = _store_prediction(conn)
    record = insert_feedback(conn, prediction.prediction_id, "billing", "low", BASE_TIME)

    with pytest.raises(FrozenInstanceError):
        record.category = "access"  # type: ignore[misc]
    row = list_current_feedback(conn)[0]
    with pytest.raises(FrozenInstanceError):
        row.category = "access"  # type: ignore[misc]


def test_fdbk_01_naive_received_at_is_rejected_without_writing(
    conn: sqlite3.Connection,
) -> None:
    prediction = _store_prediction(conn)

    with pytest.raises(ValueError):
        insert_feedback(conn, prediction.prediction_id, "billing", "low", datetime(2026, 10, 1))

    assert _feedback_rows(conn) == []
    assert not conn.in_transaction


def test_fdbk_01_invalid_label_raises_integrity_error_not_not_found(
    conn: sqlite3.Connection,
) -> None:
    prediction = _store_prediction(conn)

    with pytest.raises(sqlite3.IntegrityError):
        insert_feedback(conn, prediction.prediction_id, "not-a-category", "low", BASE_TIME)

    assert _feedback_rows(conn) == []
    assert not conn.in_transaction


def test_fdbk_02_new_feedback_keeps_previous_and_03_list_returns_latest(
    conn: sqlite3.Connection,
) -> None:
    prediction = _store_prediction(conn)
    first = insert_feedback(conn, prediction.prediction_id, "billing", "low", BASE_TIME)
    second = insert_feedback(
        conn, prediction.prediction_id, "bug", "medium", BASE_TIME + timedelta(seconds=1)
    )

    rows = _feedback_rows(conn)
    assert len(rows) == 2
    assert rows[0][1:] == (
        first.feedback_id,
        prediction.prediction_id,
        "billing",
        "low",
        to_iso(BASE_TIME),
    )
    assert rows[1][1] == second.feedback_id

    assert list_current_feedback(conn) == [
        FeedbackTrainingRow(
            prediction_id=prediction.prediction_id,
            title=prediction.title_masked,
            description=prediction.description_masked,
            category="bug",
            priority="medium",
        )
    ]


def test_fdbk_03_latest_received_at_wins_even_if_inserted_first(
    conn: sqlite3.Connection,
) -> None:
    prediction = _store_prediction(conn)
    insert_feedback(conn, prediction.prediction_id, "bug", "medium", BASE_TIME + timedelta(hours=1))
    insert_feedback(conn, prediction.prediction_id, "billing", "low", BASE_TIME)

    [row] = list_current_feedback(conn)
    assert (row.category, row.priority) == ("bug", "medium")


def test_fdbk_03_equal_received_at_highest_seq_wins(conn: sqlite3.Connection) -> None:
    prediction = _store_prediction(conn)
    insert_feedback(conn, prediction.prediction_id, "billing", "low", BASE_TIME)
    insert_feedback(conn, prediction.prediction_id, "bug", "medium", BASE_TIME)

    [row] = list_current_feedback(conn)
    assert (row.category, row.priority) == ("bug", "medium")


def test_fdbk_03_list_is_ordered_by_prediction_id_and_skips_without_feedback(
    conn: sqlite3.Connection,
) -> None:
    pred_b = _store_prediction(conn, prediction_id="bbb", title_masked="title b")
    pred_a = _store_prediction(conn, prediction_id="aaa", title_masked="title a")
    _store_prediction(conn, prediction_id="ccc")
    insert_feedback(conn, pred_b.prediction_id, "billing", "low", BASE_TIME)
    insert_feedback(conn, pred_a.prediction_id, "access", "high", BASE_TIME)

    rows = list_current_feedback(conn)

    assert [row.prediction_id for row in rows] == ["aaa", "bbb"]
    assert [row.title for row in rows] == ["title a", "title b"]


def test_fdbk_03_list_is_empty_without_feedback(conn: sqlite3.Connection) -> None:
    _store_prediction(conn)

    assert list_current_feedback(conn) == []


def test_fdbk_04_feedback_equal_to_prediction_is_stored(conn: sqlite3.Connection) -> None:
    prediction = _store_prediction(conn, category="access", priority="high")

    record = insert_feedback(conn, prediction.prediction_id, "access", "high", BASE_TIME)

    assert len(_feedback_rows(conn)) == 1
    [row] = list_current_feedback(conn)
    assert row.prediction_id == record.prediction_id
    assert (row.category, row.priority) == ("access", "high")


def test_fdbk_01_unknown_prediction_raises_not_found_and_writes_nothing(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(PredictionNotFoundError):
        insert_feedback(conn, "missing-id", "billing", "low", BASE_TIME)

    assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
    assert not conn.in_transaction


def test_fdbk_94_concurrent_feedback_same_prediction(db_path: Path) -> None:
    setup = connect(db_path)
    init_schema(setup)
    prediction = _store_prediction(setup)
    setup.close()

    count = 10
    barrier = threading.Barrier(count, timeout=30)
    errors: list[BaseException] = []
    records: list[FeedbackRecord] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            connection = connect(db_path)
            try:
                barrier.wait()
                record = insert_feedback(connection, prediction.prediction_id, "billing", "low")
                with lock:
                    records.append(record)
            finally:
                connection.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    check = connect(db_path)
    try:
        rows = check.execute(
            "SELECT seq, feedback_id, received_at FROM feedback WHERE prediction_id = ?",
            (prediction.prediction_id,),
        ).fetchall()
        # Make the latest row identifiable by giving each row distinct labels.
        latest = max(rows, key=lambda row: (row[2], row[0]))
        check.execute("UPDATE feedback SET category = 'bug' WHERE feedback_id = ?", (latest[1],))
        current = list_current_feedback(check)
    finally:
        check.close()

    assert len(rows) == count
    assert len({row[1] for row in rows}) == count
    assert {row[1] for row in rows} == {record.feedback_id for record in records}
    assert len(current) == 1
    assert current[0].category == "bug"
