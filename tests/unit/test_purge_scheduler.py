import asyncio
import logging
import sqlite3
import time
import uuid
from contextlib import suppress
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ticket_classifier.api import app as app_module
from ticket_classifier.api import purge_scheduler
from ticket_classifier.api.app import create_app
from ticket_classifier.api.purge_scheduler import run_purge_once, start_purge_loop
from ticket_classifier.settings import Settings
from ticket_classifier.storage.database import connect, init_schema, utc_now
from ticket_classifier.storage.predictions import PredictionRecord, insert_prediction

EXPECTED_LOG = "Purge finished: 1 predictions without feedback older than 90 days deleted."


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "db_path": tmp_path / "tickets.db",
        "artifacts_dir": tmp_path / "artifacts",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _store_prediction(db_path: Path, age_days: int) -> str:
    record = PredictionRecord(
        prediction_id=str(uuid.uuid4()),
        title_masked="cannot login [EMAIL]",
        description_masked="my account is locked [NUMBER]",
        category="access",
        category_confidence=0.91,
        priority="high",
        priority_confidence=0.72,
        needs_review=False,
        model_version="baseline-20260901",
        created_at=utc_now() - timedelta(days=age_days),
    )
    conn = connect(db_path)
    try:
        init_schema(conn)
        insert_prediction(conn, record)
    finally:
        conn.close()
    return record.prediction_id


def _prediction_count(db_path: Path) -> int:
    conn = connect(db_path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
    finally:
        conn.close()


def test_ops08_run_purge_once_deletes_expired_and_logs_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    _store_prediction(settings.db_path, age_days=91)
    _store_prediction(settings.db_path, age_days=10)
    caplog.set_level(logging.INFO, logger=purge_scheduler.__name__)

    deleted = asyncio.run(run_purge_once(settings))

    assert deleted == 1
    assert _prediction_count(settings.db_path) == 1
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert EXPECTED_LOG in messages


def test_ops08_run_purge_once_invalid_db_path_returns_minus_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file, not a directory")
    settings = _settings(tmp_path, db_path=blocker / "tickets.db")
    caplog.set_level(logging.INFO, logger=purge_scheduler.__name__)

    deleted = asyncio.run(run_purge_once(settings))

    assert deleted == -1
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors
    assert all(r.exc_info is None for r in errors)


def test_ops08_run_purge_once_database_error_returns_minus_one(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    # Database file exists but has no schema: the DELETE fails with sqlite3.Error.
    connect(settings.db_path).close()

    assert asyncio.run(run_purge_once(settings)) == -1


def test_ops08_run_purge_once_huge_retention_returns_minus_one(tmp_path: Path) -> None:
    settings = _settings(tmp_path, retention_days=1_000_000_000)
    _store_prediction(settings.db_path, age_days=91)

    assert asyncio.run(run_purge_once(settings)) == -1
    assert _prediction_count(settings.db_path) == 1


def test_ops08_loop_runs_immediately_and_every_interval_then_cancels_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, purge_interval_seconds=1)
    calls: list[Settings] = []

    async def fake_run(received: Settings) -> int:
        calls.append(received)
        return 0

    monkeypatch.setattr(purge_scheduler, "run_purge_once", fake_run)

    async def scenario() -> asyncio.Task[None]:
        task = start_purge_loop(settings)
        await asyncio.sleep(1.5)
        assert not task.done()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return task

    task = asyncio.run(scenario())

    assert len(calls) >= 2
    assert all(received is settings for received in calls)
    assert task.cancelled()


def test_ops08_loop_keeps_running_after_failed_purge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, purge_interval_seconds=1)
    calls: list[int] = []

    async def failing_run(received: Settings) -> int:
        calls.append(1)
        return -1

    monkeypatch.setattr(purge_scheduler, "run_purge_once", failing_run)

    async def scenario() -> None:
        task = start_purge_loop(settings)
        await asyncio.sleep(1.5)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert len(calls) >= 2


def test_ops08_app_lifespan_purges_on_startup_and_cancels_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _store_prediction(settings.db_path, age_days=91)
    started: list[asyncio.Task[None]] = []
    original = app_module.start_purge_loop

    def capturing_start(received: Settings) -> asyncio.Task[None]:
        task = original(received)
        started.append(task)
        return task

    monkeypatch.setattr(app_module, "start_purge_loop", capturing_start)

    with TestClient(create_app(settings)):
        deadline = time.monotonic() + 5.0
        while _prediction_count(settings.db_path) and time.monotonic() < deadline:
            time.sleep(0.05)

    assert _prediction_count(settings.db_path) == 0
    assert len(started) == 1
    assert started[0].done()


def test_ops08_purge_does_not_leak_connection_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    closed: list[bool] = []
    real_connect = purge_scheduler.connect

    class TrackingConnection:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def close(self) -> None:
            closed.append(True)
            self._inner.close()

    def tracking_connect(path: Path) -> TrackingConnection:
        return TrackingConnection(real_connect(path))

    def boom(*args: object, **kwargs: object) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(purge_scheduler, "connect", tracking_connect)
    monkeypatch.setattr(purge_scheduler, "purge_expired_predictions", boom)

    assert asyncio.run(run_purge_once(settings)) == -1
    assert closed == [True]
