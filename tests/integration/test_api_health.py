import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from support.api_fixtures import build_promoted_baseline
from support.sample_data import prepare_sample

from ticket_classifier.api.app import AppState, create_app
from ticket_classifier.registry import ModelRegistry
from ticket_classifier.settings import Settings
from ticket_classifier.training import train_and_register

API_KEY_WARNING = "API key not configured: /feedback will reject all requests."
UNAVAILABLE = {"status": "unavailable", "model_version": None}


@pytest.fixture(autouse=True)
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")


@pytest.fixture
def promoted_settings(tmp_path: Path) -> Settings:
    return build_promoted_baseline(tmp_path)


def _promoted_id(settings: Settings) -> str:
    promoted = ModelRegistry(settings.artifacts_dir).get_promoted()
    assert promoted is not None
    return promoted.version_id


def test_ops01_api09_health_ok_with_promoted_version_without_api_key_header(
    promoted_settings: Settings,
) -> None:
    expected_version = _promoted_id(promoted_settings)

    with TestClient(create_app(promoted_settings)) as client:
        response = client.get("/health")
        state = client.app.state.ctx  # type: ignore[attr-defined]

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": expected_version}
    assert isinstance(state, AppState)
    assert state.model_version == expected_version
    assert state.classifier is not None
    assert promoted_settings.db_path.is_file()


def test_api94_health_unavailable_without_registry(tmp_path: Path) -> None:
    settings = Settings(artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "tickets.db")

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == UNAVAILABLE


def test_api94_health_unavailable_when_promoted_artifacts_are_missing(
    promoted_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    version_id = _promoted_id(promoted_settings)
    shutil.rmtree(ModelRegistry(promoted_settings.artifacts_dir).model_dir(version_id))

    with caplog.at_level(logging.INFO), TestClient(create_app(promoted_settings)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == UNAVAILABLE
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_api94_health_unavailable_when_registry_is_corrupted(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "registry.json").write_text("{not json", encoding="utf-8")
    settings = Settings(artifacts_dir=artifacts_dir, db_path=tmp_path / "tickets.db")

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == UNAVAILABLE


def test_api09_new_promotion_is_served_only_after_new_create_app(
    tmp_path: Path, promoted_settings: Settings
) -> None:
    first_id = _promoted_id(promoted_settings)
    config = prepare_sample(tmp_path / "second_sample")
    second = train_and_register("baseline", config, promoted_settings).version

    with TestClient(create_app(promoted_settings)) as running:
        assert running.get("/health").json()["model_version"] == first_id
        ModelRegistry(promoted_settings.artifacts_dir).promote(second.version_id)
        after_promotion = running.get("/health")

    assert after_promotion.status_code == 200
    assert after_promotion.json()["model_version"] == first_id

    with TestClient(create_app(promoted_settings)) as restarted:
        restarted_response = restarted.get("/health")

    assert restarted_response.status_code == 200
    assert restarted_response.json()["model_version"] == second.version_id


def test_fdbk93_missing_api_key_logs_warning_and_health_still_works(
    promoted_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    settings = promoted_settings.model_copy(update={"api_key": None})

    with caplog.at_level(logging.INFO), TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(r.getMessage() == API_KEY_WARNING for r in warnings)


def test_fdbk93_configured_api_key_does_not_log_warning(
    promoted_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO), TestClient(create_app(promoted_settings)):
        pass

    assert all(r.getMessage() != API_KEY_WARNING for r in caplog.records)
    assert all("test-key" not in r.getMessage() for r in caplog.records)


def test_api09_importing_app_module_does_not_import_mlflow() -> None:
    code = "import sys\nimport ticket_classifier.api.app\nprint('mlflow' in sys.modules)\n"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


def test_api09_app_module_has_no_global_app_object() -> None:
    import ticket_classifier.api.app as app_module

    assert not hasattr(app_module, "app")
