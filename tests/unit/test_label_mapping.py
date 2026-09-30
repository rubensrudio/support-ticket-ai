from pathlib import Path

import pytest

from ticket_classifier.data.label_mapping import (
    LabelMapping,
    UnmappedLabelError,
    load_label_mapping,
)
from ticket_classifier.errors import PipelineError
from ticket_classifier.labels import CATEGORIES, PRIORITIES

REPO_MAPPING = Path(__file__).resolve().parents[2] / "configs" / "label_mapping.toml"

VALID_TOML = """
version = "1"
[priority]
low = "low"
[category.queue]
"Q" = "other"
"""


@pytest.fixture(scope="module")
def mapping() -> LabelMapping:
    return load_label_mapping(REPO_MAPPING)


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "mapping.toml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("queue", "tags", "expected"),
    [
        ("Billing and Payments", ["Login"], "billing"),
        ("Technical Support", ["Login"], "access"),
        ("IT Support", ["Crash"], "bug"),
        ("Product Support", ["Network"], "infrastructure"),
        ("Customer Service", ["Feedback"], "other"),
        ("Service Outages and Maintenance", ["Login"], "access"),
        ("Service Outages and Maintenance", ["Crash"], "infrastructure"),
        ("General Inquiry", [], "other"),
        ("IT Support", ["", "Bug"], "bug"),
    ],
)
def test_data02_map_category_with_repository_table(
    mapping: LabelMapping, queue: str, tags: list[str], expected: str
) -> None:
    assert mapping.map_category(queue, tags) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("very_low", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("critical", "high"),
    ],
)
def test_data02_map_priority_with_repository_table(
    mapping: LabelMapping, source: str, expected: str
) -> None:
    assert mapping.map_priority(source) == expected


def test_data02_repository_table_targets_are_canonical(mapping: LabelMapping) -> None:
    assert set(mapping.priority.values()) <= set(PRIORITIES)
    assert set(mapping.category_queue.values()) <= set(CATEGORIES)
    assert {rule.category for rule in mapping.category_rules} <= set(CATEGORIES)


def test_data91_unknown_queue_raises_unmapped_label_error(mapping: LabelMapping) -> None:
    with pytest.raises(UnmappedLabelError) as exc_info:
        mapping.map_category("Unknown Queue", [])

    assert exc_info.value.message == (
        "Unmapped source label 'Unknown Queue'. Add it to the label mapping table."
    )
    assert isinstance(exc_info.value, PipelineError)


def test_data91_unknown_queue_raises_even_when_tag_rule_matches(
    mapping: LabelMapping,
) -> None:
    with pytest.raises(UnmappedLabelError):
        mapping.map_category("Unknown Queue", ["Login"])


def test_data91_unknown_priority_raises_unmapped_label_error(mapping: LabelMapping) -> None:
    with pytest.raises(UnmappedLabelError) as exc_info:
        mapping.map_priority("urgent")

    assert "'urgent'" in exc_info.value.message
    assert exc_info.value.message == (
        "Unmapped source label 'urgent'. Add it to the label mapping table."
    )


def test_data02_first_matching_rule_wins(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        VALID_TOML
        + """
[[category.rules]]
name = "first"
any_tag_in = ["X"]
category = "bug"
[[category.rules]]
name = "second"
any_tag_in = ["X"]
category = "access"
""",
    )

    assert load_label_mapping(path).map_category("Q", ["X"]) == "bug"


def test_data02_invalid_category_target_fails_on_load(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
version = "1"
[priority]
low = "low"
[category.queue]
"Q" = "hardware"
""",
    )

    with pytest.raises(PipelineError):
        load_label_mapping(path)


def test_data02_invalid_rule_target_fails_on_load(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        VALID_TOML
        + """
[[category.rules]]
name = "r"
any_tag_in = ["X"]
category = "hardware"
""",
    )

    with pytest.raises(PipelineError):
        load_label_mapping(path)


def test_data02_invalid_priority_target_fails_on_load(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
version = "1"
[priority]
low = "urgent"
[category.queue]
"Q" = "other"
""",
    )

    with pytest.raises(PipelineError):
        load_label_mapping(path)


@pytest.mark.parametrize(
    "rule",
    [
        'name = "r"\ncategory = "bug"\n',
        'name = "r"\nqueue_in = ["Q"]\nany_tag_in = ["X"]\ncategory = "bug"\n',
        'name = "r"\nqueue_in = "Q"\ncategory = "bug"\n',
        'name = "r"\nany_tag_in = ["X"]\n',
    ],
)
def test_data02_malformed_rule_fails_on_load(tmp_path: Path, rule: str) -> None:
    path = write(tmp_path, VALID_TOML + "[[category.rules]]\n" + rule)

    with pytest.raises(PipelineError):
        load_label_mapping(path)


@pytest.mark.parametrize(
    "content",
    [
        "not = [valid toml",
        '[category.queue]\n"Q" = "other"\n',
        '[priority]\nlow = "low"\n',
        '[priority]\nlow = 1\n[category.queue]\n"Q" = "other"\n',
    ],
)
def test_data02_malformed_table_fails_on_load(tmp_path: Path, content: str) -> None:
    with pytest.raises(PipelineError):
        load_label_mapping(write(tmp_path, content))


def test_data02_missing_file_fails_on_load(tmp_path: Path) -> None:
    with pytest.raises(PipelineError):
        load_label_mapping(tmp_path / "missing.toml")
