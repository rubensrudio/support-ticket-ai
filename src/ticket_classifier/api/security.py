"""API key check for ``POST /feedback`` (AS-1, AS-7, LAC-08, LAC-20).

A single key is read from ``Settings.api_key`` (``SecretStr``). The check fails
closed: without a configured key every request is rejected. The key is compared
in constant time and must never be logged.
"""

import hmac

from ticket_classifier.settings import Settings

API_KEY_HEADER = "X-API-Key"


def is_authorized(settings: Settings, provided: str | None) -> bool:
    """Return True only if a key is configured and ``provided`` matches it exactly."""
    if settings.api_key is None or provided is None:
        return False
    expected = settings.api_key.get_secret_value()
    if not expected:
        return False
    # Compare bytes: ``compare_digest`` rejects non-ASCII ``str`` with TypeError.
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
