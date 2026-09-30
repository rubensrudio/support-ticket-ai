import pytest

from ticket_classifier.preprocessing import (
    EMAIL_MARKER,
    NUMBER_MARKER,
    PHONE_MARKER,
    URL_MARKER,
    mask_pii,
    normalize_text,
    preprocess_text,
)

CASES = [
    ("  Hi\t\tthere\n\nfriend \x07 ", "Hi there friend"),
    ("Mail john.doe@acme.com now", "Mail [EMAIL] now"),
    ("See https://acme.com/x?a=1 and www.acme.org", "See [URL] and [URL]"),
    ("Call +1 (555) 123-4567 today", "Call [PHONE] today"),
    ("Call 555-123-4567", "Call [PHONE]"),
    ("Order 12345678 failed", "Order [NUMBER] failed"),
    ("Code 12345 ok", "Code 12345 ok"),
]


def test_ct5_markers() -> None:
    assert (EMAIL_MARKER, URL_MARKER, PHONE_MARKER, NUMBER_MARKER) == (
        "[EMAIL]",
        "[URL]",
        "[PHONE]",
        "[NUMBER]",
    )


@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_data07_data08_preprocess_text(raw: str, expected: str) -> None:
    assert preprocess_text(raw) == expected


@pytest.mark.parametrize(("raw", "_expected"), CASES)
def test_data08_preprocess_text_is_idempotent(raw: str, _expected: str) -> None:
    once = preprocess_text(raw)

    assert preprocess_text(once) == once


def test_data08_normalize_replaces_control_chars_and_collapses_spaces() -> None:
    assert normalize_text("a\x00b\x1fc\x7fd\u0085e") == "a b c d e"
    assert normalize_text(" \t\r\n ") == ""
    assert normalize_text("a  b") == "a b"


def test_data08_normalize_preserves_case_and_punctuation() -> None:
    assert normalize_text("  Hello, World!  ") == "Hello, World!"


def test_data08_preprocess_equals_mask_of_normalize() -> None:
    raw = "Mail\tme at A.B@x.io\nor call 555.123.4567"

    assert preprocess_text(raw) == mask_pii(normalize_text(raw))
    assert preprocess_text(raw) == "Mail me at [EMAIL] or call [PHONE]"


def test_data07_email_masked_before_url() -> None:
    assert mask_pii("contact support@www.acme.com") == "contact [EMAIL]"


def test_data07_url_http_scheme() -> None:
    assert mask_pii("go http://x.y/z now") == "go [URL] now"


def test_data07_url_with_digits_is_not_split_into_number() -> None:
    assert mask_pii("see https://acme.com/orders/12345678") == "see [URL]"


@pytest.mark.parametrize(
    "raw",
    [
        "+5511987654321",
        "(11) 98765-4321",
        "555 123 4567",
        "+44 20 7946 0958",
    ],
)
def test_data07_phone_variants(raw: str) -> None:
    assert mask_pii(f"tel {raw} end") == "tel [PHONE] end"


def test_data07_digits_without_separator_or_plus_are_number_not_phone() -> None:
    assert mask_pii("id 5551234567") == "id [NUMBER]"


def test_data07_short_grouped_digits_are_not_phone() -> None:
    assert mask_pii("room 12-34 ok") == "room 12-34 ok"


def test_data07_lac34_grouped_digits_over_15_become_number() -> None:
    assert mask_pii("x 1234-5678-9012-3456 y") == "x [NUMBER] y"
    assert mask_pii("card 1234 5678 9012 3456") == "card [NUMBER]"
    assert mask_pii("acct 1234.5678.9012.3456.78") == "acct [NUMBER]"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Tel +55 11 98765-4321 2024", "Tel [NUMBER]"),
        ("Call 555-123-4567 123456", "Call [NUMBER]"),
        ("Pay 1234-5678-9012-3456 now", "Pay [NUMBER] now"),
    ],
)
def test_data07_lac33_lac34_no_phone_or_card_digits_leak(raw: str, expected: str) -> None:
    masked = preprocess_text(raw)

    assert masked == expected
    assert not any(char.isdigit() for char in masked)
    assert preprocess_text(masked) == masked


def test_data07_number_threshold() -> None:
    assert mask_pii("a 123456 b 12345") == "a [NUMBER] b 12345"


def test_data07_names_are_not_masked() -> None:
    assert preprocess_text("John Doe says hi") == "John Doe says hi"


def test_ct5_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        preprocess_text(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("raw", ["(555)123-4567", "555(123)4567"])
def test_data07_phone_parentheses_without_separator(raw: str) -> None:
    assert mask_pii(f"tel {raw}") == "tel [PHONE]"


def test_data07_long_digit_run_followed_by_letter_is_fast() -> None:
    assert mask_pii("1" * 5000 + "a") == "[NUMBER]a"
    assert mask_pii("12-" * 2000 + "a") == "[NUMBER]-a"
    assert mask_pii("1" * 5000 + " x") == "[NUMBER] x"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("tel555-123-4567", "tel[PHONE]"),
        ("_555-123-4567", "_[PHONE]"),
        ("Tel:555-123-4567", "Tel:[PHONE]"),
        ("x+1 555 123 4567", "x[PHONE]"),
        ("555-123-4567abc", "[PHONE]abc"),
        ("id_1234-5678-9012-3456_x", "id_[NUMBER]_x"),
    ],
)
def test_data07_phone_glued_to_word_chars_is_masked(raw: str, expected: str) -> None:
    masked = preprocess_text(raw)

    assert masked == expected
    assert preprocess_text(masked) == masked


@pytest.mark.parametrize("raw", ["v1.2.3", "build v10.20.30-rc1", "addr 0xFF12"])
def test_data07_short_versions_and_hex_are_untouched(raw: str) -> None:
    assert preprocess_text(raw) == raw


def test_data07_digits_inside_email_and_url_stay_email_and_url() -> None:
    raw = "a555-123-4567@x.com and https://x.com/555-123-4567"

    assert preprocess_text(raw) == "[EMAIL] and [URL]"
