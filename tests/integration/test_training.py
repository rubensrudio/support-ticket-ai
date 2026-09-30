import dataclasses
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from support.sample_data import prepare_sample
from support.tiny_model import build_tiny_model

from ticket_classifier import training
from ticket_classifier.cli import main
from ticket_classifier.data.splits import EmptySplitError, load_split, splits_version
from ticket_classifier.models.loader import load_classifier
from ticket_classifier.pipeline_config import PipelineConfig
from ticket_classifier.registry import ModelRegistry
from ticket_classifier.settings import Settings
from ticket_classifier.training import TrainingOutcome, train_and_register

TARGETS = ("category", "priority")
EVAL_SPLITS = ("validation", "test")
TRANSFORMER_F1_TOLERANCE = 0.01


@pytest.fixture(scope="module")
def sample_config(tmp_path_factory: pytest.TempPathFactory) -> PipelineConfig:
    tiny_dir = build_tiny_model(tmp_path_factory.mktemp("tiny_model"))
    return prepare_sample(tmp_path_factory.mktemp("sample"), model_name=str(tiny_dir))


@pytest.fixture(autouse=True)
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(artifacts_dir=tmp_path / "artifacts")


def _pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["title"], frame["description"], strict=True))


def _probability_rows(outcome_classifier: Any, frame: pd.DataFrame) -> list[Any]:
    results = outcome_classifier.predict_proba(
        frame["title"].tolist(), frame["description"].tolist()
    )
    return [(r.category, r.priority) for r in results]


def _test_macro_f1(outcome: TrainingOutcome) -> dict[str, float]:
    return {target: outcome.version.metrics["test"][target]["macro_f1"] for target in TARGETS}


def test_model08_baseline_registers_version_with_metrics_and_metadata(
    sample_config: PipelineConfig, settings: Settings
) -> None:
    outcome = train_and_register("baseline", sample_config, settings)

    version = outcome.version
    registry = ModelRegistry(settings.artifacts_dir)
    stored = registry.get(version.version_id)
    assert outcome.promoted is False
    assert stored.status == "registered"
    assert stored.kind == "baseline"
    assert registry.get_promoted() is None
    assert stored.splits_version == splits_version(sample_config.data.processed_dir)
    assert stored.seed == sample_config.seed
    assert set(stored.hyperparameters) == set(TARGETS)
    assert stored.hyperparameters["category"]["C"] in sample_config.baseline.c_grid
    for split in EVAL_SPLITS:
        for target in TARGETS:
            metrics = stored.metrics[split][target]
            assert 0.0 <= metrics["macro_f1"] <= 1.0
            assert "accuracy" in metrics
            assert "confusion_matrix" in metrics
    train_rows = len(load_split(sample_config.data.processed_dir, "train"))
    assert stored.training_rows == train_rows
    assert stored.feedback_rows == 0
    assert stored.artifact_dir == f"models/{version.version_id}"
    assert (settings.artifacts_dir / stored.artifact_dir).is_dir()


def test_model06_baseline_retraining_gives_identical_test_metrics(
    sample_config: PipelineConfig, settings: Settings
) -> None:
    first = train_and_register("baseline", sample_config, settings)
    second = train_and_register("baseline", sample_config, settings)

    assert first.version.version_id != second.version.version_id
    assert first.version.metrics["test"] == second.version.metrics["test"]


def test_model09_first_transformer_is_promoted_second_is_not(
    sample_config: PipelineConfig, settings: Settings
) -> None:
    first = train_and_register("transformer", sample_config, settings)
    second = train_and_register("transformer", sample_config, settings)

    registry = ModelRegistry(settings.artifacts_dir)
    assert first.promoted is True
    assert first.version.status == "promoted"
    assert second.promoted is False
    assert registry.get(second.version.version_id).status == "registered"
    promoted = registry.get_promoted()
    assert promoted is not None
    assert promoted.version_id == first.version.version_id
    assert "best_epoch" in first.version.hyperparameters

    # MODEL-06: same splits, seed and config -> Macro F1 within 1 percentage point.
    first_f1 = _test_macro_f1(first)
    second_f1 = _test_macro_f1(second)
    for target in TARGETS:
        assert abs(first_f1[target] - second_f1[target]) <= TRANSFORMER_F1_TOLERANCE


def test_model09_transformer_with_extra_train_is_not_promoted(
    sample_config: PipelineConfig, settings: Settings
) -> None:
    extra = load_split(sample_config.data.processed_dir, "validation").head(5)

    outcome = train_and_register("transformer", sample_config, settings, extra_train=extra)

    assert outcome.promoted is False
    assert ModelRegistry(settings.artifacts_dir).get_promoted() is None
    train_rows = len(load_split(sample_config.data.processed_dir, "train"))
    assert outcome.version.training_rows == train_rows + 5
    assert outcome.version.feedback_rows == 5


def test_model90_empty_test_split_fails_before_training(
    sample_config: PipelineConfig,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "processed"
    shutil.copytree(sample_config.data.processed_dir, processed)
    (processed / "test.jsonl").write_text("", encoding="utf-8")
    config = dataclasses.replace(
        sample_config, data=dataclasses.replace(sample_config.data, processed_dir=processed)
    )
    calls: list[str] = []
    monkeypatch.setattr(training, "train_baseline", lambda *args, **kwargs: calls.append("trained"))

    with pytest.raises(EmptySplitError) as excinfo:
        train_and_register("baseline", config, settings)

    assert excinfo.value.split == "test"
    assert calls == []
    assert ModelRegistry(settings.artifacts_dir).list_versions() == []
    assert not (settings.artifacts_dir / "models").exists()


@pytest.mark.parametrize(
    ("kind", "trainer_name"),
    [("baseline", "train_baseline"), ("transformer", "train_transformer")],
)
def test_model07_trainers_never_receive_test_records(
    sample_config: PipelineConfig,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    trainer_name: str,
) -> None:
    real_trainer = getattr(training, trainer_name)
    received: list[pd.DataFrame] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        received.extend(
            value for value in (*args, *kwargs.values()) if isinstance(value, pd.DataFrame)
        )
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(training, trainer_name, spy)

    train_and_register(kind, sample_config, settings)  # type: ignore[arg-type]

    test_pairs = _pairs(load_split(sample_config.data.processed_dir, "test"))
    assert len(received) == 2
    for frame in received:
        assert _pairs(frame).isdisjoint(test_pairs)


@pytest.mark.parametrize("kind", ["baseline", "transformer"])
def test_model08_load_classifier_matches_trained_model(
    sample_config: PipelineConfig,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    trained: list[Any] = []
    trainer_name = f"train_{kind}"
    real_trainer = getattr(training, trainer_name)

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = real_trainer(*args, **kwargs)
        trained.append(result[0])
        return result

    monkeypatch.setattr(training, trainer_name, capture)
    outcome = train_and_register(kind, sample_config, settings)  # type: ignore[arg-type]

    loaded = load_classifier(outcome.version, settings.artifacts_dir)

    batch = load_split(sample_config.data.processed_dir, "validation").head(10)
    assert loaded.kind == kind
    expected = _probability_rows(trained[0], batch)
    actual = _probability_rows(loaded, batch)
    for (exp_cat, exp_pri), (act_cat, act_pri) in zip(expected, actual, strict=True):
        assert act_cat == pytest.approx(exp_cat, abs=1e-6)
        assert act_pri == pytest.approx(exp_pri, abs=1e-6)


def test_ops04_cli_train_baseline_registers_version(
    sample_config: PipelineConfig,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TICKET_ARTIFACTS_DIR", str(settings.artifacts_dir))
    config_path = sample_config.data.processed_dir.parent / "pipeline.toml"

    exit_code = main(["train-baseline", "--config", str(config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Registered model version '" in out
    assert "(baseline). Promoted: no." in out
    assert len(ModelRegistry(settings.artifacts_dir).list_versions()) == 1


def test_model90_cli_reports_empty_split(
    sample_config: PipelineConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = sample_config.data.processed_dir.parent / "pipeline.toml"
    content = config_path.read_text(encoding="utf-8").replace(
        str(sample_config.data.processed_dir.as_posix()), (tmp_path / "missing").as_posix()
    )
    other_config = tmp_path / "pipeline.toml"
    other_config.write_text(content, encoding="utf-8")
    monkeypatch.setenv("TICKET_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    exit_code = main(["train-transformer", "--config", str(other_config)])

    assert exit_code == 1
    assert "Split 'train' is empty or missing" in capsys.readouterr().err
