"""Text preprocessing shared by training and inference (DATA-07, DATA-08, DA-6).

``preprocess_text`` is the single routine applied to ticket text in both the
training pipeline and the API, so the same input always yields the same output.
It first normalizes whitespace/control characters and then masks PII in a fixed
order: EMAIL -> URL -> PHONE -> NUMBER. Letter case and punctuation are preserved.
"""

import re
import unicodedata

EMAIL_MARKER = "[EMAIL]"
URL_MARKER = "[URL]"
PHONE_MARKER = "[PHONE]"
NUMBER_MARKER = "[NUMBER]"

_WHITESPACE_RE = re.compile(r"\s+")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

# Candidate phone: optional "+", then digit groups (optionally in parentheses)
# joined by one to three separators among space, "." and "-". A parenthesized
# group may touch its neighbours without a separator. Two plain digit runs never
# touch, which keeps matching linear (no catastrophic backtracking). The digit
# count and the separator/plus requirement are checked in ``_replace_phone``;
# candidates over 15 digits become ``[NUMBER]``.
_DIGIT_GROUP = r"(?:\(\d+\)|\d+)"
_PHONE_NEXT_GROUP = rf"(?:[ .\-]{{1,3}}{_DIGIT_GROUP}|\(\d+\)|(?<=\))\d+)"
_PHONE_CANDIDATE_RE = re.compile(
    rf"(?<![\w+])(?P<plus>\+ ?)?{_DIGIT_GROUP}{_PHONE_NEXT_GROUP}*(?!\w)"
)
_PHONE_MIN_DIGITS = 8
_PHONE_MAX_DIGITS = 15

_NUMBER_RE = re.compile(r"\d{6,}")


def _check_str(text: object) -> None:
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")


def normalize_text(text: str) -> str:
    """Replace Unicode control characters (``Cc``) with spaces, collapse
    whitespace runs into a single space and strip both ends."""
    _check_str(text)
    without_controls = "".join(" " if unicodedata.category(char) == "Cc" else char for char in text)
    return _WHITESPACE_RE.sub(" ", without_controls).strip()


def _replace_phone(match: re.Match[str]) -> str:
    """Mask a phone candidate.

    A candidate with more than 15 digits (a phone followed by more digits, or a
    card/account number written in groups) is masked whole as ``[NUMBER]`` so
    no phone or long identifier leaks (LAC-33, LAC-34).
    """
    candidate = match.group(0)
    digit_count = sum(char.isdigit() for char in candidate)
    if digit_count > _PHONE_MAX_DIGITS:
        return NUMBER_MARKER
    if digit_count < _PHONE_MIN_DIGITS:
        return candidate
    has_plus = match.group("plus") is not None
    has_separator = any(char in " .-()" for char in candidate)
    return PHONE_MARKER if has_plus or has_separator else candidate


def mask_pii(text: str) -> str:
    """Mask e-mails, URLs, phone numbers and runs of 6+ digits, in that order.

    Digit groups joined by space, "." or "-" totalling more than 15 digits are
    masked as ``[NUMBER]``.
    """
    _check_str(text)
    masked = _EMAIL_RE.sub(EMAIL_MARKER, text)
    masked = _URL_RE.sub(URL_MARKER, masked)
    masked = _PHONE_CANDIDATE_RE.sub(_replace_phone, masked)
    return _NUMBER_RE.sub(NUMBER_MARKER, masked)


def preprocess_text(text: str) -> str:
    """Normalize then mask PII. Idempotent: ``preprocess_text(preprocess_text(x))``
    equals ``preprocess_text(x)``."""
    return mask_pii(normalize_text(text))
