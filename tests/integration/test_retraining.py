import dataclasses
import fcntl
import multiprocessing
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from support.sample_data import prepare_sample
from support.tiny_model import build_tiny_model

from ticket_classifier import retraining
from ticket_classifier.cli import main
from ticket_classifier.data.splits import load_split, splits_version
from ticket_classifier.errors import PipelineError
from ticket_classifier.pipeline_config import PipelineConfig
from ticket_classifier.registry import ModelRegistry
from ticket_classifier.retraining import (
    NoFeedbackError,
    RetrainInProgressError,
    RetrainResult,
    run_retraining,
)
from ticket_classifier.settings import Settings
from ticket_classifier.storage.database import connect, init_schema
from ticket_classifier.storage.feedback import insert_feedback
from ticket_classifier.storage.predictions import PredictionRecord, insert_prediction
from ticket_classifier.training import TrainingOutcome, train_and_register

TARGETS = ("category", "priority")
LOCK_WAIT_SECONDS = 30

# prediction_id -> (masked title, masked description)
PREDICTIONS: dict[str, tuple[str, str]] = {
    "pred-a": ("fb title a [EMAIL]", "fb description a [PHONE]"),
    "pred-b": ("fb title b", "fb description b [EMAIL]"),
    "pred-c": ("fb title c", "fb description c"),
}
# (prediction_id, category, priority, minutes after base time); pred-a gets two.
FEEDBACKS: tuple[tuple[str, str, str, int], ...] = (
    ("pred-a", "bug", "low", 0),
    ("pred-a", "billing", "high", 5),
    ("pred-b", "access", "medium", 1),
    ("pred-c", "other", "low", 2),
)
EXPECTED_ROWS: set[tuple[str, str, str, str]] = {
    ("fb title a [EMAIL]", "fb description a [PHONE]", "billing", "high"),
    ("fb title b", "fb description b [EMAIL]", "access", "medium"),
    ("fb title c", "fb description c", "other", "low"),
}


@pytest.fixture(scope="module")
def sample_config(tmp_path_factory: pytest.TempPathFactory) -> PipelineConfig:
    tiny_dir = build_tiny_model(tmp_path_factory.mktemp("tiny_model"))
    return prepare_sample(tmp_path_factory.mktemp("sample"), model_name=str(tiny_dir))


@pytest.fixture(scope="module")
def promoted_artifacts(
    sample_config: PipelineConfig, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Artifacts dir with one promoted transformer version, trained once per module."""
    root = tmp_path_factory.mktemp("promoted")
    mp = pytest.MonkeyPatch()
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{root}/mlflow.db")
    try:
        outcome = train_and_register(
            "transformer", sample_config, Settings(artifacts_dir=root / "artifacts")
        )
    finally:
        mp.undo()
    assert outcome.promoted is True
    return root / "artifacts"


@pytest.fixture(autouse=True)
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")


@pytest.fixture
def settings(tmp_path: Path, promoted_artifacts: Path) -> Settings:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(promoted_artifacts, artifacts)
    return Settings(artifacts_dir=artifacts, db_path=tmp_path / "var" / "tickets.db")


@pytest.fixture
def with_feedback(settings: Settings) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    conn = connect(settings.db_path)
    try:
        init_schema(conn)
        for prediction_id, (title, description) in PREDICTIONS.items():
            insert_prediction(
                conn,
                PredictionRecord(
                    prediction_id=prediction_id,
                    title_masked=title,
                    description_masked=description,
                    category="other",
                    category_confidence=0.5,
                    priority="medium",
                    priority_confidence=0.5,
                    needs_review=True,
                    model_version="transformer-20260101T000000Z-00000000",
                    created_at=base,
                ),
            )
        for prediction_id, category, priority, minutes in FEEDBACKS:
            insert_feedback(
                conn, prediction_id, category, priority, base + timedelta(minutes=minutes)
            )
    finally:
        conn.close()


def _promoted_id(settings: Settings) -> str:
    promoted = ModelRegistry(settings.artifacts_dir).get_promoted()
    assert promoted is not None
    return promoted.version_id


def _force_test_f1(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, delta: float
) -> list[pd.DataFrame]:
    """Wrap ``train_and_register``: set the new version's test Macro F1 to promoted + delta."""
    promoted = ModelRegistry(settings.artifacts_dir).get_promoted()
    assert promoted is not None
    real = retraining.train_and_register
    received: list[pd.DataFrame] = []

    def wrapper(*args: Any, **kwargs: Any) -> TrainingOutcome:
        extra = kwargs.get("extra_train")
        if isinstance(extra, pd.DataFrame):
            received.append(extra.copy())
        outcome = real(*args, **kwargs)
        for target in TARGETS:
            baseline_f1 = promoted.test_macro_f1(target)
            assert baseline_f1 is not None
            outcome.version.metrics["test"][target]["macro_f1"] = baseline_f1 + delta
        return outcome

    monkeypatch.setattr(retraining, "train_and_register", wrapper)
    return received


@pytest.mark.usefixtures("with_feedback")
def test_retr01_extra_train_has_current_feedback_with_masked_text(
    sample_config: PipelineConfig, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    received = _force_test_f1(monkeypatch, settings, 0.0)

    result = run_retraining(sample_config, settings)

    assert len(received) == 1
    extra = received[0]
    assert len(extra) == 3
    rows = set(
        zip(extra["title"], extra["description"], extra["category"], extra["priority"], strict=True)
    )
    assert rows == EXPECTED_ROWS
    assert result.feedback_rows == 3
    assert result.new_version.kind == "transformer"
    assert result.new_version.feedback_rows == 3


@pytest.mark.usefixtures("with_feedback")
def test_retr02_test_split_is_the_promoted_one_without_feedback(
    sample_config: PipelineConfig, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous_id = _promoted_id(settings)
    _force_test_f1(monkeypatch, settings, -0.01)

    result = run_retraining(sample_config, settings)

    data_dir = sample_config.data.processed_dir
    assert result.previous_version.version_id == previous_id
    assert result.new_version.splits_version == result.previous_version.splits_version
    assert result.new_version.splits_version == splits_version(data_dir)
    test_texts = set(load_split(data_dir, "test")["title"]) | set(
        load_split(data_dir, "test")["description"]
    )
    for title, description in PREDICTIONS.values():
        assert title not in test_texts
        assert description not in test_texts


def test_retr02_changed_splits_abort_before_training(
    sample_config: PipelineConfig,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry(settings.artifacts_dir)
    before = registry.list_versions()
    processed = tmp_path / "processed"
    shutil.copytree(sample_config.data.processed_dir, processed)
    with (processed / "test.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"title": "x", "description": "y", "category": "bug", "priority": "low"}\n')
    config = dataclasses.replace(
        sample_config, data=dataclasses.replace(sample_config.data, processed_dir=processed)
    )
    calls: list[str] = []
    monkeypatch.setattr(retraining, "train_and_register", lambda *a, **k: calls.append("x"))

    with pytest.raises(PipelineError) as excinfo:
        run_retraining(config, settings)

    current = splits_version(processed)
    expected = before[0].splits_version
    assert excinfo.value.message == (
        f"Frozen test set changed: splits version '{current}' differs from promoted "
        f"version '{expected}'. Retraining aborted."
    )
    assert calls == []
    assert registry.list_versions() == before


@pytest.mark.usefixtures("with_feedback")
def test_retr03_better_or_equal_metrics_promote_and_retire_previous(
    sample_config: PipelineConfig, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous_id = _promoted_id(settings)
    _force_test_f1(monkeypatch, settings, 0.0)

    result = run_retraining(sample_config, settings)

    registry = ModelRegistry(settings.artifacts_dir)
    assert result.decision == "promoted"
    assert registry.get(result.new_version.version_id).status == "promoted"
    assert registry.get(previous_id).status == "retired"
    assert result.new_version.status == "promoted"


@pytest.mark.usefixtures("with_feedback")
def test_retr03_worse_metrics_register_without_promoting(
    sample_config: PipelineConfig, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous_id = _promoted_id(settings)
    _force_test_f1(monkeypatch, settings, -0.01)

    result = run_retraining(sample_config, settings)

    registry = ModelRegistry(settings.artifacts_dir)
    assert result.decision == "not promoted"
    assert registry.get(result.new_version.version_id).status == "registered"
    assert registry.get(previous_id).status == "promoted"
    assert _promoted_id(settings) == previous_id


@pytest.mark.parametrize("init_db", [True, False])
def test_retr90_no_feedback_aborts_without_new_version(
    sample_config: PipelineConfig,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    init_db: bool,
) -> None:
    if init_db:
        conn = connect(settings.db_path)
        try:
            init_schema(conn)
        finally:
            conn.close()
    registry = ModelRegistry(settings.artifacts_dir)
    before = registry.list_versions()
    calls: list[str] = []
    monkeypatch.setattr(retraining, "train_and_register", lambda *a, **k: calls.append("x"))

    with pytest.raises(NoFeedbackError) as excinfo:
        run_retraining(sample_config, settings)

    assert excinfo.value.message == "No feedback records available. Retraining aborted."
    assert calls == []
    assert registry.list_versions() == before


def test_retr90_without_promoted_version_aborts(
    sample_config: PipelineConfig, tmp_path: Path
) -> None:
    settings = Settings(artifacts_dir=tmp_path / "empty", db_path=tmp_path / "tickets.db")

    with pytest.raises(PipelineError) as excinfo:
        run_retraining(sample_config, settings)

    assert excinfo.value.message == "No promoted model version. Train a transformer first."


@pytest.mark.usefixtures("with_feedback")
def test_retr91_training_failure_keeps_promoted_version(
    sample_config: PipelineConfig, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous_id = _promoted_id(settings)
    registry = ModelRegistry(settings.artifacts_dir)
    before = registry.list_versions()

    def boom(*args: Any, **kwargs: Any) -> TrainingOutcome:
        raise RuntimeError("training exploded")

    monkeypatch.setattr(retraining, "train_and_register", boom)

    with pytest.raises(RuntimeError):
        run_retraining(sample_config, settings)

    after = registry.list_versions()
    assert _promoted_id(settings) == previous_id
    assert [v.status for v in after].count("promoted") == 1
    assert after == before
    # The lock is released after a failure: a second run is not refused as concurrent.
    with pytest.raises(RuntimeError):
        run_retraining(sample_config, settings)


def _hold_lock(lock_path: str, acquired: Any, release: Any, result: Any) -> None:
    with open(lock_path, "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired.set()
        release.wait(LOCK_WAIT_SECONDS)
        result.value = 1
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@pytest.mark.usefixtures("with_feedback")
def test_retr92_concurrent_run_is_refused(
    sample_config: PipelineConfig, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(retraining, "train_and_register", lambda *a, **k: calls.append("x"))
    lock_path = settings.artifacts_dir / "retrain.lock"
    ctx = multiprocessing.get_context("fork")
    acquired, release = ctx.Event(), ctx.Event()
    finished = ctx.Value("i", 0)
    holder = ctx.Process(target=_hold_lock, args=(str(lock_path), acquired, release, finished))
    holder.start()
    try:
        assert acquired.wait(LOCK_WAIT_SECONDS)

        with pytest.raises(RetrainInProgressError) as excinfo:
            run_retraining(sample_config, settings)

        assert excinfo.value.message == "A retraining run is already in progress."
        assert calls == []
        assert holder.is_alive()
        with open(lock_path, "a") as probe, pytest.raises(BlockingIOError):
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        release.set()
        holder.join(LOCK_WAIT_SECONDS)
    assert holder.exitcode == 0
    assert finished.value == 1


@pytest.mark.usefixtures("with_feedback")
def test_retr04_cli_prints_decision_and_metrics(
    sample_config: PipelineConfig,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TICKET_ARTIFACTS_DIR", str(settings.artifacts_dir))
    monkeypatch.setenv("TICKET_DB_PATH", str(settings.db_path))
    config_path = sample_config.data.processed_dir.parent / "pipeline.toml"
    previous_id = _promoted_id(settings)

    exit_code = main(["retrain", "--config", str(config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Retraining finished: 3 feedback records used. Decision: " in out
    decision_line = next(line for line in out.splitlines() if "Decision: " in line)
    assert decision_line.endswith(("Decision: promoted.", "Decision: not promoted."))
    assert previous_id in out
    for target in TARGETS:
        assert f"test_{target}_macro_f1" in out


def test_retr90_cli_without_feedback_exits_1(
    sample_config: PipelineConfig,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TICKET_ARTIFACTS_DIR", str(settings.artifacts_dir))
    monkeypatch.setenv("TICKET_DB_PATH", str(settings.db_path))
    config_path = sample_config.data.processed_dir.parent / "pipeline.toml"

    exit_code = main(["retrain", "--config", str(config_path)])

    assert exit_code == 1
    assert "No feedback records available. Retraining aborted." in capsys.readouterr().err


def test_retrain_result_is_a_dataclass() -> None:
    assert RetrainResult.__dataclass_fields__.keys() == {
        "feedback_rows",
        "new_version",
        "previous_version",
        "decision",
    }
