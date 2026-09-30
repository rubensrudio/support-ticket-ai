import json
import re
from pathlib import Path
from typing import Any

import pytest

from ticket_classifier.cli import main
from ticket_classifier.errors import PipelineError
from ticket_classifier.registry import ModelRegistry, ModelVersion

VERSION_ID_PATTERN = re.compile(r"^baseline-\d{8}T\d{6}Z-[0-9a-f]{8}$")


def _metrics(category_f1: float, priority_f1: float) -> dict[str, dict[str, dict[str, Any]]]:
    def target(f1: float) -> dict[str, Any]:
        return {
            "labels": ["a", "b"],
            "accuracy": 0.5,
            "macro_f1": f1,
            "per_class": {},
            "confusion_matrix": [[1, 0], [0, 1]],
        }

    return {
        "validation": {"category": target(0.1), "priority": target(0.2)},
        "test": {"category": target(category_f1), "priority": target(priority_f1)},
    }


def _version(
    version_id: str,
    created_at: str,
    kind: str = "baseline",
    status: str = "registered",
) -> ModelVersion:
    return ModelVersion(
        version_id=version_id,
        kind=kind,
        status=status,
        created_at=created_at,
        splits_version="splits-abc",
        seed=42,
        hyperparameters={"C": 1.0},
        metrics=_metrics(0.81234, 0.65432),
        training_rows=100,
        artifact_dir=f"models/{version_id}",
        feedback_rows=0,
    )


@pytest.fixture()
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path)


def test_model08_new_version_id_matches_pattern_and_is_unique(registry: ModelRegistry) -> None:
    first = registry.new_version_id("baseline")
    second = registry.new_version_id("baseline")

    assert VERSION_ID_PATTERN.match(first)
    assert VERSION_ID_PATTERN.match(second)
    assert first != second


def test_model08_new_version_id_rejects_unknown_kind(registry: ModelRegistry) -> None:
    with pytest.raises(PipelineError):
        registry.new_version_id("../evil")


def test_model_dir_is_under_models(registry: ModelRegistry, tmp_path: Path) -> None:
    version_id = registry.new_version_id("transformer")

    assert registry.model_dir(version_id) == tmp_path / "models" / version_id


def test_model_dir_rejects_path_traversal(registry: ModelRegistry) -> None:
    with pytest.raises(PipelineError):
        registry.model_dir("../outside")


def test_model08_register_and_get_roundtrip(registry: ModelRegistry) -> None:
    version = _version("baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00")

    registry.register(version)

    assert registry.get(version.version_id) == version


def test_model08_to_dict_from_dict_roundtrip() -> None:
    version = _version("baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00")

    assert ModelVersion.from_dict(version.to_dict()) == version


def test_register_duplicate_raises(registry: ModelRegistry) -> None:
    version = _version("baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00")
    registry.register(version)

    with pytest.raises(PipelineError):
        registry.register(version)


def test_register_requires_registered_status(registry: ModelRegistry) -> None:
    version = _version(
        "baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00", status="promoted"
    )

    with pytest.raises(PipelineError):
        registry.register(version)


def test_get_unknown_raises(registry: ModelRegistry) -> None:
    with pytest.raises(PipelineError):
        registry.get("baseline-20260101T000000Z-0000000a")


def test_ops05_list_versions_orders_by_created_at(registry: ModelRegistry) -> None:
    later = _version("baseline-20260102T000000Z-0000000b", "2026-01-02T00:00:00+00:00")
    earlier = _version("baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00")
    registry.register(later)
    registry.register(earlier)

    ids = [v.version_id for v in registry.list_versions()]

    assert ids == [earlier.version_id, later.version_id]


def test_ops06_promote_retires_previous(registry: ModelRegistry) -> None:
    a = _version(
        "transformer-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00", "transformer"
    )
    b = _version(
        "transformer-20260102T000000Z-0000000b", "2026-01-02T00:00:00+00:00", "transformer"
    )
    registry.register(a)
    registry.register(b)

    registry.promote(a.version_id)
    registry.promote(b.version_id)

    assert registry.get(a.version_id).status == "retired"
    assert registry.get(b.version_id).status == "promoted"
    assert [v.status for v in registry.list_versions()].count("promoted") == 1
    promoted = registry.get_promoted()
    assert promoted is not None
    assert promoted.version_id == b.version_id


def test_ops06_promote_retired_raises_and_keeps_file(
    registry: ModelRegistry, tmp_path: Path
) -> None:
    a = _version(
        "transformer-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00", "transformer"
    )
    b = _version(
        "transformer-20260102T000000Z-0000000b", "2026-01-02T00:00:00+00:00", "transformer"
    )
    registry.register(a)
    registry.register(b)
    registry.promote(a.version_id)
    registry.promote(b.version_id)
    before = (tmp_path / "registry.json").read_bytes()

    with pytest.raises(PipelineError):
        registry.promote(a.version_id)

    assert (tmp_path / "registry.json").read_bytes() == before


def test_ops06_promote_promoted_raises(registry: ModelRegistry) -> None:
    a = _version(
        "transformer-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00", "transformer"
    )
    registry.register(a)
    registry.promote(a.version_id)

    with pytest.raises(PipelineError):
        registry.promote(a.version_id)


def test_ops06_promote_unknown_raises_and_keeps_file(
    registry: ModelRegistry, tmp_path: Path
) -> None:
    registry.register(_version("baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00"))
    before = (tmp_path / "registry.json").read_bytes()

    with pytest.raises(PipelineError):
        registry.promote("baseline-20260109T000000Z-0000000f")

    assert (tmp_path / "registry.json").read_bytes() == before


def test_get_promoted_without_registry_file_returns_none(registry: ModelRegistry) -> None:
    assert registry.get_promoted() is None
    assert registry.list_versions() == []


def test_latest_returns_most_recent_of_kind(registry: ModelRegistry) -> None:
    registry.register(_version("baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00"))
    registry.register(_version("baseline-20260103T000000Z-0000000c", "2026-01-03T00:00:00+00:00"))
    registry.register(
        _version(
            "transformer-20260105T000000Z-0000000d", "2026-01-05T00:00:00+00:00", "transformer"
        )
    )

    latest = registry.latest("baseline")

    assert latest is not None
    assert latest.version_id == "baseline-20260103T000000Z-0000000c"
    assert registry.latest("transformer") is not None
    assert ModelRegistry(registry.artifacts_dir / "empty").latest("baseline") is None


def test_registry_file_layout(registry: ModelRegistry, tmp_path: Path) -> None:
    version = _version("baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00")
    registry.register(version)

    data = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))

    assert data == {"versions": [version.to_dict()]}
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupted_registry_raises_pipeline_error(tmp_path: Path) -> None:
    (tmp_path / "registry.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(PipelineError):
        ModelRegistry(tmp_path).list_versions()


def test_ops05_list_versions_cli_prints_columns(
    registry: ModelRegistry, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _version("baseline-20260101T000000Z-0000000a", "2026-01-01T00:00:00+00:00")
    b = _version(
        "transformer-20260102T000000Z-0000000b", "2026-01-02T00:00:00+00:00", "transformer"
    )
    registry.register(a)
    registry.register(b)
    registry.promote(b.version_id)

    exit_code = main(["list-versions", "--artifacts-dir", str(tmp_path)])

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert exit_code == 0
    header = lines[0].split()
    assert header == [
        "version_id",
        "kind",
        "test_category_macro_f1",
        "test_priority_macro_f1",
        "created_at",
        "promoted",
    ]
    assert lines[1].split() == [
        a.version_id,
        "baseline",
        "0.8123",
        "0.6543",
        "2026-01-01T00:00:00+00:00",
        "no",
    ]
    assert lines[2].split() == [
        b.version_id,
        "transformer",
        "0.8123",
        "0.6543",
        "2026-01-02T00:00:00+00:00",
        "yes",
    ]


def test_ops05_list_versions_cli_empty_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["list-versions", "--artifacts-dir", str(tmp_path)])

    assert exit_code == 0
    assert "version_id" in capsys.readouterr().out
