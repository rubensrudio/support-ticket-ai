import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ticket_classifier.cli import main
from ticket_classifier.storage.database import connect, init_schema
from ticket_classifier.storage.feedback import PredictionNotFoundError, insert_feedback
from ticket_classifier.storage.predictions import PredictionRecord, insert_prediction
from ticket_classifier.storage.purge import PURGE_MESSAGE, purge_expired_predictions

NOW = datetime(2026, 10, 1, 12, 0, 0, 123456, tzinfo=UTC)


def _prediction(created_at: datetime) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=str(uuid.uuid4()),
        title_masked="cannot login [EMAIL]",
        description_masked="my account is locked [NUMBER]",
        category="access",
        category_confidence=0.91,
        priority="high",
        priority_confidence=0.72,
        needs_review=False,
        model_version="baseline-20260901",
        created_at=created_at,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "tickets.db"


@pytest.fixture
def conn(db_path: Path) -> sqlite3.Connection:
    connection = connect(db_path)
    init_schema(connection)
    yield connection  # type: ignore[misc]
    connection.close()


def _store(conn: sqlite3.Connection, age_days: int) -> PredictionRecord:
    record = _prediction(NOW - timedelta(days=age_days))
    insert_prediction(conn, record)
    return record


def _prediction_ids(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT prediction_id FROM predictions").fetchall()}


def _feedback_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])


def _orphan_feedback_count(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM feedback AS f "
            "LEFT JOIN predictions AS p ON p.prediction_id = f.prediction_id "
            "WHERE p.prediction_id IS NULL"
        ).fetchone()[0]
    )


def test_ops07_purge_deletes_only_expired_predictions_without_feedback(
    conn: sqlite3.Connection,
) -> None:
    age_89 = _store(conn, 89)
    age_90 = _store(conn, 90)
    age_91 = _store(conn, 91)
    age_91_with_feedback = _store(conn, 91)
    insert_feedback(conn, age_91_with_feedback.prediction_id, "billing", "low", NOW)

    deleted = purge_expired_predictions(conn, NOW)

    assert deleted == 1
    assert _prediction_ids(conn) == {
        age_89.prediction_id,
        age_90.prediction_id,
        age_91_with_feedback.prediction_id,
    }
    assert age_91.prediction_id not in _prediction_ids(conn)


def test_ops07_purge_never_deletes_feedback(conn: sqlite3.Connection) -> None:
    for age in (10, 91, 200):
        record = _store(conn, age)
        insert_feedback(conn, record.prediction_id, "billing", "low", NOW)
        insert_feedback(conn, record.prediction_id, "access", "high", NOW)
    _store(conn, 120)
    before = _feedback_count(conn)

    deleted = purge_expired_predictions(conn, NOW)

    assert deleted == 1
    assert _feedback_count(conn) == before == 6
    assert _orphan_feedback_count(conn) == 0


def test_ops07_purge_respects_custom_retention_days(conn: sqlite3.Connection) -> None:
    recent = _store(conn, 5)
    old = _store(conn, 11)

    deleted = purge_expired_predictions(conn, NOW, retention_days=10)

    assert deleted == 1
    assert _prediction_ids(conn) == {recent.prediction_id}
    assert old.prediction_id not in _prediction_ids(conn)


def test_ops07_purge_on_empty_database_returns_zero(conn: sqlite3.Connection) -> None:
    assert purge_expired_predictions(conn, NOW) == 0


def test_ops07_purge_leaves_no_open_transaction(conn: sqlite3.Connection) -> None:
    _store(conn, 91)

    purge_expired_predictions(conn, NOW)

    assert conn.in_transaction is False


def test_ops07_purge_rejects_non_positive_retention(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        purge_expired_predictions(conn, NOW, retention_days=0)


def test_ops07_purge_rejects_naive_now(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        purge_expired_predictions(conn, datetime(2026, 10, 1, 12, 0, 0))


def test_ops07_purge_message_format() -> None:
    assert PURGE_MESSAGE.format(count=3, days=90) == (
        "Purge finished: 3 predictions without feedback older than 90 days deleted."
    )


class _HoldingConnection:
    """Proxy that keeps the purge transaction open after the DELETE runs."""

    def __init__(self, inner: sqlite3.Connection, holding: threading.Event) -> None:
        self._inner = inner
        self._holding = holding

    def execute(self, sql: str, *params: Any) -> sqlite3.Cursor:
        cursor = self._inner.execute(sql, *params)
        if sql.lstrip().upper().startswith("DELETE"):
            self._holding.set()
            time.sleep(0.5)
        return cursor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_ops90_feedback_during_purge_never_leaves_orphan_feedback(
    conn: sqlite3.Connection, db_path: Path
) -> None:
    target = _store(conn, 91)
    purge_conn = connect(db_path)
    feedback_conn = connect(db_path)
    holding = threading.Event()
    purge_result: list[int] = []
    feedback_outcome: list[object] = []

    def run_purge() -> None:
        proxy = _HoldingConnection(purge_conn, holding)
        purge_result.append(purge_expired_predictions(proxy, NOW))  # type: ignore[arg-type]

    def run_feedback() -> None:
        assert holding.wait(timeout=5)
        try:
            feedback_outcome.append(
                insert_feedback(feedback_conn, target.prediction_id, "billing", "low", NOW)
            )
        except PredictionNotFoundError as exc:
            feedback_outcome.append(exc)

    thread_a = threading.Thread(target=run_purge)
    thread_b = threading.Thread(target=run_feedback)
    try:
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=10)
        thread_b.join(timeout=10)
    finally:
        purge_conn.close()
        feedback_conn.close()

    assert len(purge_result) == 1
    assert len(feedback_outcome) == 1
    prediction_kept = target.prediction_id in _prediction_ids(conn)
    feedback_stored = _feedback_count(conn) == 1
    if prediction_kept:
        assert feedback_stored
    else:
        assert isinstance(feedback_outcome[0], PredictionNotFoundError)
        assert _feedback_count(conn) == 0
    assert _orphan_feedback_count(conn) == 0


def test_ops90_feedback_before_purge_keeps_prediction(conn: sqlite3.Connection) -> None:
    target = _store(conn, 91)
    insert_feedback(conn, target.prediction_id, "billing", "low", NOW)

    assert purge_expired_predictions(conn, NOW) == 0
    assert target.prediction_id in _prediction_ids(conn)
    assert _orphan_feedback_count(conn) == 0


def test_ops07_cli_purge_prints_message_and_returns_zero(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TICKET_DB_PATH", str(db_path))
    monkeypatch.delenv("TICKET_RETENTION_DAYS", raising=False)
    seed = connect(db_path)
    try:
        init_schema(seed)
        insert_prediction(seed, _prediction(datetime.now(UTC) - timedelta(days=120)))
        insert_prediction(seed, _prediction(datetime.now(UTC) - timedelta(days=1)))
    finally:
        seed.close()

    exit_code = main(["purge"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "Purge finished: 1 predictions without feedback older than 90 days deleted."
    )


def test_ops07_cli_purge_on_new_database_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TICKET_DB_PATH", str(tmp_path / "var" / "fresh.db"))
    monkeypatch.delenv("TICKET_RETENTION_DAYS", raising=False)

    exit_code = main(["purge"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "Purge finished: 0 predictions without feedback older than 90 days deleted."
    )


def test_ops07_cli_purge_database_error_returns_one_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a sqlite database" * 100)
    monkeypatch.setenv("TICKET_DB_PATH", str(broken))

    exit_code = main(["purge"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    assert captured.err.strip() != ""
