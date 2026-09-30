import pytest
from pydantic import SecretStr

from ticket_classifier.api.security import API_KEY_HEADER, is_authorized
from ticket_classifier.settings import Settings

CONFIGURED_KEY = "s3cret-key"


def _settings(api_key: str | None) -> Settings:
    return Settings(api_key=SecretStr(api_key) if api_key is not None else None)


def test_fdbk90_header_name_is_x_api_key() -> None:
    assert API_KEY_HEADER == "X-API-Key"


def test_fdbk01_matching_key_is_authorized() -> None:
    assert is_authorized(_settings(CONFIGURED_KEY), CONFIGURED_KEY) is True


@pytest.mark.parametrize(
    "provided",
    [None, "", "wrong", CONFIGURED_KEY + "x", CONFIGURED_KEY[:-1], CONFIGURED_KEY.upper()],
)
def test_fdbk90_missing_or_different_key_is_rejected(provided: str | None) -> None:
    assert is_authorized(_settings(CONFIGURED_KEY), provided) is False


def test_fdbk90_non_ascii_key_is_rejected_without_error() -> None:
    assert is_authorized(_settings(CONFIGURED_KEY), "clé-ü") is False


def test_fdbk90_non_ascii_configured_key_matches_itself() -> None:
    assert is_authorized(_settings("clé"), "clé") is True


@pytest.mark.parametrize("provided", [None, "", "anything", "None"])
def test_fdbk93_no_configured_key_rejects_everything(provided: str | None) -> None:
    assert is_authorized(_settings(None), provided) is False


def test_fdbk93_blank_configured_key_rejects_blank_provided_key() -> None:
    settings = Settings.model_construct(api_key=SecretStr(""))

    assert is_authorized(settings, "") is False


def test_fdbk90_uses_constant_time_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    from ticket_classifier.api import security

    calls: list[tuple[bytes, bytes]] = []

    def fake_compare(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return a == b

    monkeypatch.setattr(security.hmac, "compare_digest", fake_compare)

    assert is_authorized(_settings(CONFIGURED_KEY), CONFIGURED_KEY) is True
    assert len(calls) == 1
