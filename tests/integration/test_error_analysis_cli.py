import json
from pathlib import Path

import pytest
from support.sample_data import (
    make_source_rows,
    prepare_sample,
    write_pipeline_config,
    write_source_csv,
)

from ticket_classifier.cli import main
from ticket_classifier.registry import ModelRegistry
from ticket_classifier.settings import Settings
from ticket_classifier.training import train_and_register

NO_PROMOTED_MESSAGE = "No promoted model version. Train a transformer first."


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("TICKET_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("TICKET_REVIEW_THRESHOLD", "0.6")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    return artifacts_dir


def test_model10_analyze_errors_without_promoted_version_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_pipeline_config(
        tmp_path, write_source_csv(tmp_path / "raw" / "tickets.csv", make_source_rows(5))
    )

    code = main(["analyze-errors", "--config", str(config_path)])

    assert code == 1
    assert NO_PROMOTED_MESSAGE in capsys.readouterr().err


def test_model10_model11_analyze_errors_with_promoted_baseline_writes_reports(
    tmp_path: Path, isolated_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = prepare_sample(tmp_path)
    settings = Settings(artifacts_dir=isolated_env)
    outcome = train_and_register("baseline", config, settings)
    registry = ModelRegistry(isolated_env)
    registry.promote(outcome.version.version_id)
    version_id = outcome.version.version_id

    code = main(["analyze-errors", "--config", str(tmp_path / "pipeline.toml")])

    assert code == 0
    md_path = isolated_env / "reports" / f"error_analysis_{version_id}.md"
    json_path = isolated_env / "reports" / f"error_analysis_{version_id}.json"
    assert md_path.is_file()
    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["version_id"] == version_id
    assert data["threshold"] == 0.6
    assert 0.0 <= data["needs_review_fraction"] <= 1.0
    assert len(data["top_category_confusions"]) <= 5
    assert version_id in capsys.readouterr().out


def test_model10_analyze_errors_with_explicit_version(tmp_path: Path, isolated_env: Path) -> None:
    config = prepare_sample(tmp_path)
    outcome = train_and_register("baseline", config, Settings(artifacts_dir=isolated_env))
    version_id = outcome.version.version_id

    code = main(
        [
            "analyze-errors",
            "--version",
            version_id,
            "--config",
            str(tmp_path / "pipeline.toml"),
        ]
    )

    assert code == 0
    assert (isolated_env / "reports" / f"error_analysis_{version_id}.json").is_file()
