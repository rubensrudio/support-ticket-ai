import os
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from ticket_classifier.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_ticket_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("TICKET_"):
            monkeypatch.delenv(name, raising=False)


def test_api04_defaults_without_ticket_env() -> None:
    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.review_threshold == 0.6
    assert settings.api_key is None
    assert settings.db_path == Path("var/tickets.db")
    assert settings.artifacts_dir == Path("artifacts")
    assert settings.retention_days == 90
    assert settings.purge_interval_seconds == 86400


def test_api04_review_threshold_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TICKET_REVIEW_THRESHOLD", "0.3")

    assert get_settings().review_threshold == 0.3


@pytest.mark.parametrize("value", ["0", "1"])
def test_api04_review_threshold_accepts_bounds(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TICKET_REVIEW_THRESHOLD", value)

    assert get_settings().review_threshold == float(value)


@pytest.mark.parametrize("value", ["1.5", "-0.1"])
def test_api04_review_threshold_out_of_range_raises(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TICKET_REVIEW_THRESHOLD", value)

    with pytest.raises(ValidationError):
        get_settings()


@pytest.mark.parametrize("value", ["", "   "])
def test_fdbk93_blank_api_key_is_none(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TICKET_API_KEY", value)

    assert get_settings().api_key is None


def test_fdbk93_api_key_is_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TICKET_API_KEY", "s3cr3t")

    settings = get_settings()

    assert isinstance(settings.api_key, SecretStr)
    assert "s3cr3t" not in repr(settings)
    assert "s3cr3t" not in str(settings)
    assert settings.api_key.get_secret_value() == "s3cr3t"


def test_path_and_int_fields_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TICKET_DB_PATH", "/tmp/x.db")
    monkeypatch.setenv("TICKET_ARTIFACTS_DIR", "models")
    monkeypatch.setenv("TICKET_RETENTION_DAYS", "30")
    monkeypatch.setenv("TICKET_PURGE_INTERVAL_SECONDS", "60")

    settings = get_settings()

    assert settings.db_path == Path("/tmp/x.db")
    assert settings.artifacts_dir == Path("models")
    assert settings.retention_days == 30
    assert settings.purge_interval_seconds == 60


@pytest.mark.parametrize("name", ["TICKET_RETENTION_DAYS", "TICKET_PURGE_INTERVAL_SECONDS"])
def test_positive_int_fields_reject_zero(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValidationError):
        get_settings()


def test_get_settings_returns_new_instance_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    monkeypatch.setenv("TICKET_REVIEW_THRESHOLD", "0.9")
    second = get_settings()

    assert first is not second
    assert first.review_threshold == 0.6
    assert second.review_threshold == 0.9
