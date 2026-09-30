import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ticket_classifier.data.splits import (
    SPLIT_NAMES,
    EmptySplitError,
    load_split,
    splits_version,
    write_splits,
)
from ticket_classifier.errors import PipelineError

COLUMNS = ["title", "description", "category", "priority"]


def make_frame(prefix: str, rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [f"{prefix} title {i} café" for i in range(rows)],
            "description": [f"{prefix} description {i}\nline two" for i in range(rows)],
            "category": ["billing" if i % 2 else "technical" for i in range(rows)],
            "priority": ["high" if i % 3 else "low" for i in range(rows)],
        },
        columns=COLUMNS,
    )


@pytest.fixture
def splits() -> dict[str, pd.DataFrame]:
    return {
        "train": make_frame("tr", 5),
        "validation": make_frame("va", 3),
        "test": make_frame("te", 2),
    }


@pytest.fixture
def report() -> dict[str, Any]:
    return {"seed": 42, "total_source_rows": 10}


def test_split_names_contract() -> None:
    assert SPLIT_NAMES == ("train", "validation", "test")


def test_empty_split_error_is_pipeline_error() -> None:
    assert issubclass(EmptySplitError, PipelineError)


def test_round_trip_preserves_rows_columns_and_order(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    out = tmp_path / "processed"
    write_splits(splits, report, out)
    for name in SPLIT_NAMES:
        assert_frame_equal(load_split(out, name), splits[name])


def test_round_trip_ignores_extra_columns_and_non_default_index(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    shuffled = splits["train"].iloc[::-1].assign(extra=1)
    write_splits({**splits, "train": shuffled}, report, tmp_path)
    expected = shuffled[COLUMNS].reset_index(drop=True)
    assert_frame_equal(load_split(tmp_path, "train"), expected)


def test_writes_jsonl_utf8_one_record_per_line(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    write_splits(splits, report, tmp_path)
    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert "café" in lines[0]
    assert list(json.loads(lines[0]).keys()) == COLUMNS


def test_two_writes_are_deterministic(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    v1 = write_splits(splits, report, first)
    v2 = write_splits(splits, report, second)
    assert v1 == v2
    for filename in ["train.jsonl", "validation.jsonl", "test.jsonl", "prepare_report.json"]:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_version_is_sha256_prefix_of_concatenated_splits(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    version = write_splits(splits, report, tmp_path)
    data = b"".join((tmp_path / f"{n}.jsonl").read_bytes() for n in SPLIT_NAMES)
    assert version == hashlib.sha256(data).hexdigest()[:12]
    assert splits_version(tmp_path) == version


def test_version_changes_when_data_changes(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    v1 = write_splits(splits, report, tmp_path / "a")
    v2 = write_splits({**splits, "test": make_frame("other", 2)}, report, tmp_path / "b")
    assert v1 != v2


def test_report_contains_splits_version(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    version = write_splits(splits, report, tmp_path)
    saved = json.loads((tmp_path / "prepare_report.json").read_text(encoding="utf-8"))
    assert saved["splits_version"] == version
    assert saved["seed"] == 42
    assert "splits_version" not in report


def test_rewrite_replaces_previous_files_and_leaves_no_temp_dir(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    out = tmp_path / "processed"
    write_splits(splits, report, out)
    new_train = make_frame("new", 1)
    write_splits({**splits, "train": new_train}, report, out)
    assert_frame_equal(load_split(out, "train"), new_train)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["processed"]


def test_write_rejects_missing_split(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    del splits["test"]
    with pytest.raises(ValueError):
        write_splits(splits, report, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_write_rejects_missing_column(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    splits["validation"] = splits["validation"].drop(columns=["priority"])
    with pytest.raises(ValueError):
        write_splits(splits, report, tmp_path / "out")


def test_model_90_load_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(EmptySplitError) as exc_info:
        load_split(tmp_path, "validation")
    assert exc_info.value.message == (
        "Split 'validation' is empty or missing. Run data preparation first."
    )


def test_model_90_load_empty_split_raises(tmp_path: Path) -> None:
    (tmp_path / "validation.jsonl").write_bytes(b"")
    with pytest.raises(EmptySplitError) as exc_info:
        load_split(tmp_path, "validation")
    assert str(exc_info.value) == (
        "Split 'validation' is empty or missing. Run data preparation first."
    )


def test_model_90_empty_dataframe_written_then_load_raises(
    tmp_path: Path, splits: dict[str, pd.DataFrame], report: dict[str, Any]
) -> None:
    write_splits({**splits, "test": make_frame("te", 0)}, report, tmp_path)
    with pytest.raises(EmptySplitError, match="Split 'test'"):
        load_split(tmp_path, "test")


def test_load_rejects_unknown_split_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_split(tmp_path, "holdout")
