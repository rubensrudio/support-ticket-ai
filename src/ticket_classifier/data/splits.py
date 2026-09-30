"""Persistence and versioning of the train/validation/test splits (DATA-01, DATA-06, MODEL-90).

Format (plan section 7.2):

- ``<split>.jsonl``: one JSON object per line with ``title``, ``description``,
  ``category`` and ``priority`` (``ensure_ascii=False``, fixed order).
- ``prepare_report.json``: preparation report plus ``splits_version``.
- ``splits_version``: first 12 hex chars of the SHA-256 of the bytes of
  ``train.jsonl``, ``validation.jsonl`` and ``test.jsonl`` concatenated in that order.
"""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ticket_classifier.errors import PipelineError

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")
SPLIT_COLUMNS: tuple[str, str, str, str] = ("title", "description", "category", "priority")
REPORT_FILENAME = "prepare_report.json"

_VERSION_LENGTH = 12


class EmptySplitError(PipelineError):
    """A required split is empty or missing (MODEL-90)."""

    def __init__(self, split: str) -> None:
        super().__init__(f"Split '{split}' is empty or missing. Run data preparation first.")
        self.split: str = split


def _split_path(data_dir: Path, name: str) -> Path:
    return data_dir / f"{name}.jsonl"


def _check_split_name(name: str) -> None:
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split name: {name!r}")


def _serialize_split(name: str, frame: pd.DataFrame) -> bytes:
    missing = [column for column in SPLIT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Split {name!r} is missing columns: {missing}")
    lines: list[str] = []
    for row in frame.loc[:, list(SPLIT_COLUMNS)].itertuples(index=False, name=None):
        record = {column: str(value) for column, value in zip(SPLIT_COLUMNS, row, strict=True)}
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    return "".join(lines).encode("utf-8")


def _version_of(contents: list[bytes]) -> str:
    digest = hashlib.sha256()
    for content in contents:
        digest.update(content)
    return digest.hexdigest()[:_VERSION_LENGTH]


def write_splits(splits: Mapping[str, pd.DataFrame], report: dict[str, Any], out_dir: Path) -> str:
    """Write the three splits and the report atomically per file; return ``splits_version``.

    Files are first written to a temporary sibling directory and then moved into
    ``out_dir`` with ``os.replace``. The given ``report`` is not mutated.
    """
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"Expected splits {list(SPLIT_NAMES)}, got {sorted(splits)}")

    contents = [_serialize_split(name, splits[name]) for name in SPLIT_NAMES]
    version = _version_of(contents)
    full_report = {**report, "splits_version": version}
    report_bytes = (json.dumps(full_report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    files: dict[str, bytes] = {
        _split_path(Path(), name).name: content
        for name, content in zip(SPLIT_NAMES, contents, strict=True)
    }
    files[REPORT_FILENAME] = report_bytes

    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        for filename, content in files.items():
            with open(tmp_dir / filename, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            os.replace(tmp_dir / filename, out_dir / filename)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return version


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    """Load a split as a DataFrame with ``title, description, category, priority``.

    Raises ``EmptySplitError`` when the file is missing or has no records (MODEL-90).
    """
    _check_split_name(name)
    path = _split_path(Path(data_dir), name)
    if not path.is_file():
        raise EmptySplitError(name)
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise EmptySplitError(name)
    return pd.DataFrame(
        {column: [str(record[column]) for record in records] for column in SPLIT_COLUMNS},
        columns=list(SPLIT_COLUMNS),
    )


def splits_version(data_dir: Path) -> str:
    """Return the ``splits_version`` of the split files stored in ``data_dir``."""
    contents: list[bytes] = []
    for name in SPLIT_NAMES:
        path = _split_path(Path(data_dir), name)
        if not path.is_file():
            raise EmptySplitError(name)
        contents.append(path.read_bytes())
    return _version_of(contents)
