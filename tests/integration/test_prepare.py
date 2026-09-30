import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from support.sample_data import (
    CATEGORY_SOURCES,
    LABEL_MAPPING_PATH,
    SAMPLE_PER_CATEGORY,
    make_source_rows,
    prepare_sample,
    write_pipeline_config,
    write_source_csv,
)

from ticket_classifier.cli import main
from ticket_classifier.data.label_mapping import UnmappedLabelError
from ticket_classifier.data.prepare import (
    InsufficientExamplesError,
    SourceDatasetNotFoundError,
    prepare_dataset,
)
from ticket_classifier.data.splits import REPORT_FILENAME, SPLIT_NAMES
from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.pipeline_config import PipelineConfig, load_pipeline_config

SPLIT_KEYS = {"title", "description", "category", "priority"}


def _read_split(directory: Path, name: str) -> list[dict[str, Any]]:
    lines = (directory / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _config_for(tmp_path: Path, rows: list[dict[str, str]]) -> PipelineConfig:
    source = write_source_csv(tmp_path / "raw" / "tickets.csv", rows)
    return load_pipeline_config(write_pipeline_config(tmp_path, source))


def _assert_no_files(directory: Path) -> None:
    assert not directory.exists() or not any(directory.iterdir())


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> PipelineConfig:
    return prepare_sample(tmp_path_factory.mktemp("prepared"))


def _report(config: PipelineConfig) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(
        (config.data.processed_dir / REPORT_FILENAME).read_text(encoding="utf-8")
    )
    return report


def test_data01_splits_have_exactly_the_four_keys(prepared: PipelineConfig) -> None:
    for name in SPLIT_NAMES:
        records = _read_split(prepared.data.processed_dir, name)
        assert records
        for record in records:
            assert set(record) == SPLIT_KEYS
            assert all(isinstance(value, str) and value for value in record.values())
            assert record["category"] in CATEGORIES
            assert record["priority"] in PRIORITIES


def test_data03_data09_data90_data11_report_counts(prepared: PipelineConfig) -> None:
    report = _report(prepared)

    assert report["discarded"] == {"non_english": 10, "empty_text": 3, "duplicate": 5}
    assert report["total_source_rows"] == SAMPLE_PER_CATEGORY * len(CATEGORY_SOURCES) + 18
    assert report["seed"] == prepared.seed
    assert len(report["source_sha256"]) == 64
    assert report["source_path"] == str(prepared.data.source_path)
    assert len(report["splits_version"]) == 12
    for name in SPLIT_NAMES:
        records = _read_split(prepared.data.processed_dir, name)
        split = report["splits"][name]
        assert split["total"] == len(records)
        assert split["category"] == {
            label: sum(r["category"] == label for r in records) for label in CATEGORIES
        }
        assert split["priority"] == {
            label: sum(r["priority"] == label for r in records) for label in PRIORITIES
        }


def test_prepare_dataset_returns_the_report(tmp_path: Path) -> None:
    config = _config_for(tmp_path, make_source_rows())

    report = prepare_dataset(config, LABEL_MAPPING_PATH)

    assert report == _report(config)


def test_data04_data05_split_sizes_and_stratification(prepared: PipelineConfig) -> None:
    splits = {name: _read_split(prepared.data.processed_dir, name) for name in SPLIT_NAMES}
    total = sum(len(records) for records in splits.values())
    assert total == SAMPLE_PER_CATEGORY * len(CATEGORY_SOURCES)

    expected = {"train": 0.70, "validation": 0.15, "test": 0.15}
    all_categories = Counter(r["category"] for records in splits.values() for r in records)
    for name, records in splits.items():
        assert abs(len(records) - expected[name] * total) <= 1
        counts = Counter(r["category"] for r in records)
        for category in CATEGORIES:
            overall = all_categories[category] / total
            assert abs(counts[category] / len(records) - overall) <= 0.01


def test_data06_same_seed_is_byte_identical_and_other_seed_differs(tmp_path: Path) -> None:
    config = _config_for(tmp_path, make_source_rows())
    first = dataclasses.replace(
        config, data=dataclasses.replace(config.data, processed_dir=tmp_path / "first")
    )
    second = dataclasses.replace(
        config, data=dataclasses.replace(config.data, processed_dir=tmp_path / "second")
    )
    other = dataclasses.replace(
        config,
        seed=config.seed + 1,
        data=dataclasses.replace(config.data, processed_dir=tmp_path / "other"),
    )

    for item in (first, second, other):
        prepare_dataset(item, LABEL_MAPPING_PATH)

    for name in SPLIT_NAMES:
        filename = f"{name}.jsonl"
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
    assert (tmp_path / "first" / "test.jsonl").read_bytes() != (
        tmp_path / "other" / "test.jsonl"
    ).read_bytes()


def test_data07_data08_pii_is_masked_in_written_text(prepared: PipelineConfig) -> None:
    content = "".join(
        (prepared.data.processed_dir / f"{name}.jsonl").read_text(encoding="utf-8")
        for name in SPLIT_NAMES
    )

    assert "@" not in content
    assert "555-123-4567" not in content
    assert "[EMAIL]" in content
    assert "[PHONE]" in content
    assert "\\\\n" not in content
    assert "\\n" not in content


def test_data92_missing_source_raises_and_writes_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "raw" / "absent.csv"
    config = load_pipeline_config(write_pipeline_config(tmp_path, missing))

    with pytest.raises(SourceDatasetNotFoundError) as exc_info:
        prepare_dataset(config, LABEL_MAPPING_PATH)

    assert exc_info.value.message == (
        f"Source dataset not found at '{missing}'. See README for download instructions."
    )
    _assert_no_files(config.data.processed_dir)


def test_data92_unreadable_source_raises_and_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "tickets.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"\xff\xfe\x00not-utf8\xff")
    config = load_pipeline_config(write_pipeline_config(tmp_path, source))

    with pytest.raises(SourceDatasetNotFoundError):
        prepare_dataset(config, LABEL_MAPPING_PATH)

    _assert_no_files(config.data.processed_dir)


def test_data91_unmapped_queue_raises_and_writes_nothing(tmp_path: Path) -> None:
    rows = make_source_rows()
    rows[0]["queue"] = "Unknown Queue"
    config = _config_for(tmp_path, rows)

    with pytest.raises(UnmappedLabelError):
        prepare_dataset(config, LABEL_MAPPING_PATH)

    _assert_no_files(config.data.processed_dir)


def test_data10_insufficient_access_examples_raises_and_writes_nothing(tmp_path: Path) -> None:
    access_queue, access_tag = CATEGORY_SOURCES["access"]
    rows: list[dict[str, str]] = []
    access_kept = 0
    for row in make_source_rows():
        is_access = row["queue"] == access_queue and row["tag_1"] == access_tag
        if is_access and row["language"] == "en" and row["subject"]:
            if access_kept >= 60:
                continue
            access_kept += 1
        rows.append(row)
    config = _config_for(tmp_path, rows)

    with pytest.raises(InsufficientExamplesError) as exc_info:
        prepare_dataset(config, LABEL_MAPPING_PATH)

    message = exc_info.value.message
    assert message.startswith("Category 'access' has ")
    assert message.endswith(" training examples; minimum is 100.")
    count = int(message.removeprefix("Category 'access' has ").split(" ")[0])
    assert 0 < count < 100
    _assert_no_files(config.data.processed_dir)


def test_data10_missing_category_raises(tmp_path: Path) -> None:
    access_queue, access_tag = CATEGORY_SOURCES["access"]
    rows = [
        row
        for row in make_source_rows()
        if not (row["queue"] == access_queue and row["tag_1"] == access_tag)
    ]
    config = _config_for(tmp_path, rows)

    with pytest.raises(InsufficientExamplesError) as exc_info:
        prepare_dataset(config, LABEL_MAPPING_PATH)

    assert exc_info.value.message == "Category 'access' has 0 training examples; minimum is 100."
    _assert_no_files(config.data.processed_dir)


def test_cli_prepare_returns_zero(tmp_path: Path) -> None:
    source = write_source_csv(tmp_path / "raw" / "tickets.csv", make_source_rows())
    config_path = write_pipeline_config(tmp_path, source)

    exit_code = main(
        ["prepare", "--config", str(config_path), "--mapping", str(LABEL_MAPPING_PATH)]
    )

    assert exit_code == 0
    for name in SPLIT_NAMES:
        assert (tmp_path / "processed" / f"{name}.jsonl").is_file()


def test_cli_prepare_default_mapping_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source_csv(tmp_path / "raw" / "tickets.csv", make_source_rows())
    config_path = write_pipeline_config(tmp_path, source)
    monkeypatch.chdir(LABEL_MAPPING_PATH.parents[1])

    assert main(["prepare", "--config", str(config_path)]) == 0


def test_cli_prepare_missing_source_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "raw" / "absent.csv"
    config_path = write_pipeline_config(tmp_path, missing)

    exit_code = main(
        ["prepare", "--config", str(config_path), "--mapping", str(LABEL_MAPPING_PATH)]
    )

    assert exit_code == 1
    assert capsys.readouterr().err.strip() == (
        f"Source dataset not found at '{missing}'. See README for download instructions."
    )
    _assert_no_files(tmp_path / "processed")
