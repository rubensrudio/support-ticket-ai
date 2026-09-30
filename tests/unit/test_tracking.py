import json
from pathlib import Path
from typing import Any

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from ticket_classifier.registry import ModelRegistry, ModelVersion
from ticket_classifier.tracking import flatten_params, log_training_run

CATEGORY_LABELS = ["access", "billing"]
PRIORITY_LABELS = ["low", "medium", "high"]


def _target(labels: list[str], macro_f1: float) -> dict[str, Any]:
    size = len(labels)
    return {
        "labels": labels,
        "accuracy": 0.75,
        "macro_f1": macro_f1,
        "per_class": {
            label: {"precision": 0.5, "recall": 0.6, "f1": 0.55, "support": 4} for label in labels
        },
        "confusion_matrix": [[1 if i == j else 0 for j in range(size)] for i in range(size)],
    }


def _version(tmp_path: Path) -> ModelVersion:
    registry = ModelRegistry(tmp_path / "artifacts")
    version_id = registry.new_version_id("baseline")
    return ModelVersion(
        version_id=version_id,
        kind="baseline",
        status="registered",
        created_at="2026-09-30T12:00:00+00:00",
        splits_version="splits-abc123",
        seed=42,
        hyperparameters={"tfidf": {"max_features": 5000, "ngram_range": [1, 2]}, "C": 1.0},
        metrics={
            split: {
                "category": _target(CATEGORY_LABELS, 0.8),
                "priority": _target(PRIORITY_LABELS, 0.6),
            }
            for split in ("validation", "test")
        },
        training_rows=100,
        artifact_dir=f"models/{version_id}",
        feedback_rows=3,
    )


@pytest.fixture
def tracking_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    return tmp_path


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "model_files"
    directory.mkdir()
    (directory / "baseline_meta.json").write_text("{}", encoding="utf-8")
    return directory


def test_ops04_run_has_params_and_metrics(tracking_env: Path, model_dir: Path) -> None:
    version = _version(tracking_env)

    run_id = log_training_run(version, model_dir)

    mlflow.set_tracking_uri(f"sqlite:///{tracking_env}/mlflow.db")
    run = mlflow.get_run(run_id)
    params = run.data.params
    assert params["seed"] == "42"
    assert params["splits_version"] == "splits-abc123"
    assert params["version_id"] == version.version_id
    assert params["kind"] == "baseline"
    assert params["feedback_rows"] == "3"
    assert params["tfidf.max_features"] == "5000"
    assert params["C"] == "1.0"

    metrics = run.data.metrics
    assert metrics["test_category_macro_f1"] == pytest.approx(0.8)
    assert metrics["validation_priority_macro_f1"] == pytest.approx(0.6)
    assert metrics["test_category_accuracy"] == pytest.approx(0.75)
    assert metrics["test_category_access_precision"] == pytest.approx(0.5)
    assert metrics["test_priority_high_recall"] == pytest.approx(0.6)
    assert metrics["validation_category_billing_f1"] == pytest.approx(0.55)

    experiment = mlflow.get_experiment(run.info.experiment_id)
    assert experiment.name == "support-ticket-classifier"


def test_ops04_run_has_confusion_matrix_and_model_artifacts(
    tracking_env: Path, model_dir: Path
) -> None:
    version = _version(tracking_env)

    run_id = log_training_run(version, model_dir, experiment_name="custom-exp")

    client = MlflowClient(tracking_uri=f"sqlite:///{tracking_env}/mlflow.db")
    top_level = {item.path for item in client.list_artifacts(run_id)}
    assert "confusion_matrix_test_category.json" in top_level
    assert "confusion_matrix_validation_priority.json" in top_level
    assert "model" in top_level
    model_files = {item.path for item in client.list_artifacts(run_id, "model")}
    assert "model/baseline_meta.json" in model_files

    local = Path(
        client.download_artifacts(run_id, "confusion_matrix_test_category.json", str(tracking_env))
    )
    payload = json.loads(local.read_text(encoding="utf-8"))
    assert payload["labels"] == CATEGORY_LABELS
    assert payload["matrix"] == [[1, 0], [0, 1]]


def test_ops04_missing_model_dir_raises(tracking_env: Path) -> None:
    from ticket_classifier.errors import PipelineError

    with pytest.raises(PipelineError):
        log_training_run(_version(tracking_env), tracking_env / "missing")


def test_flatten_params_nested_and_lists() -> None:
    flat = flatten_params({"a": {"b": 1, "c": {"d": "x"}}, "e": [1, 2], "f": None})
    assert flat == {"a.b": "1", "a.c.d": "x", "e": "[1, 2]", "f": "None"}
