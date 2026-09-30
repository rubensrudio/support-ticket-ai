import json
from pathlib import Path
from typing import Any

import pytest

from ticket_classifier.cli import main
from ticket_classifier.errors import PipelineError
from ticket_classifier.evaluation.metrics import ClassMetrics, TargetMetrics
from ticket_classifier.evaluation.report import (
    ComparisonReport,
    build_comparison,
    render_markdown,
    write_report,
)
from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.registry import ModelRegistry, ModelVersion


def _target_metrics(labels: tuple[str, ...], macro_f1: float, seed: int) -> dict[str, Any]:
    size = len(labels)
    return TargetMetrics(
        labels=list(labels),
        accuracy=0.5 + seed / 100,
        macro_f1=macro_f1,
        per_class={
            label: ClassMetrics(
                precision=0.1 * (index + 1) + seed / 1000,
                recall=0.05 * (index + 1) + seed / 1000,
                f1=0.07 * (index + 1) + seed / 1000,
                support=10 + index,
            )
            for index, label in enumerate(labels)
        },
        confusion_matrix=[[seed + row * size + col for col in range(size)] for row in range(size)],
    ).to_dict()


def _version(
    version_id: str,
    kind: str,
    category_f1: float,
    priority_f1: float,
    splits_version: str = "splits-abc",
    created_at: str = "2026-01-01T00:00:00+00:00",
    seed: int = 1,
) -> ModelVersion:
    test = {
        "category": _target_metrics(CATEGORIES, category_f1, seed),
        "priority": _target_metrics(PRIORITIES, priority_f1, seed),
    }
    return ModelVersion(
        version_id=version_id,
        kind=kind,
        status="registered",
        created_at=created_at,
        splits_version=splits_version,
        seed=42,
        hyperparameters={},
        metrics={"validation": test, "test": test},
        training_rows=100,
        artifact_dir=f"models/{version_id}",
    )


BASELINE_ID = "baseline-20260101T000000Z-aaaaaaaa"
TRANSFORMER_ID = "transformer-20260102T000000Z-bbbbbbbb"


def _baseline(**kwargs: Any) -> ModelVersion:
    return _version(BASELINE_ID, "baseline", 0.60, 0.50, seed=1, **kwargs)


def _transformer(category_f1: float, priority_f1: float, **kwargs: Any) -> ModelVersion:
    return _version(
        TRANSFORMER_ID,
        "transformer",
        category_f1,
        priority_f1,
        seed=2,
        created_at="2026-01-02T00:00:00+00:00",
        **kwargs,
    )


def test_model05_pass_when_category_gains_5pp_and_priority_does_not_drop() -> None:
    report = build_comparison(_baseline(), _transformer(0.65, 0.50))

    assert report.verdict == "PASS"
    assert report.macro_f1_delta_pp == {"category": 5.0, "priority": 0.0}
    assert report.baseline_id == BASELINE_ID
    assert report.transformer_id == TRANSFORMER_ID


@pytest.mark.parametrize(("category_f1", "priority_f1"), [(0.649, 0.60), (0.70, 0.49)])
def test_model05_fail_when_a_threshold_is_missed(category_f1: float, priority_f1: float) -> None:
    report = build_comparison(_baseline(), _transformer(category_f1, priority_f1))

    assert report.verdict == "FAIL"


def test_model04_delta_is_rounded_percentage_points() -> None:
    report = build_comparison(_baseline(), _transformer(0.6512345, 0.4))

    assert report.macro_f1_delta_pp == {"category": 5.12, "priority": -10.0}


def test_model04_metrics_hold_both_models_test_metrics_per_target() -> None:
    baseline = _baseline()
    transformer = _transformer(0.65, 0.50)

    report = build_comparison(baseline, transformer)

    for target in ("category", "priority"):
        assert report.metrics[target]["baseline"] == baseline.metrics["test"][target]
        assert report.metrics[target]["transformer"] == transformer.metrics["test"][target]


def test_model04_different_splits_raise_pipeline_error() -> None:
    with pytest.raises(PipelineError) as exc_info:
        build_comparison(_baseline(), _transformer(0.65, 0.5, splits_version="splits-xyz"))

    assert exc_info.value.message == "Model versions were evaluated on different splits."


def test_model04_missing_test_metrics_raise_pipeline_error() -> None:
    transformer = _transformer(0.65, 0.5)
    transformer.metrics = {"validation": transformer.metrics["validation"]}

    with pytest.raises(PipelineError):
        build_comparison(_baseline(), transformer)


@pytest.mark.parametrize(("category_f1", "verdict"), [(0.65, "PASS"), (0.60, "FAIL")])
def test_model04_markdown_has_verdict_matrices_and_per_class_rows(
    category_f1: float, verdict: str
) -> None:
    baseline = _baseline()
    transformer = _transformer(category_f1, 0.50)
    report = build_comparison(baseline, transformer)

    markdown = render_markdown(report)

    assert f"Verdict: {verdict}" in markdown
    assert BASELINE_ID in markdown
    assert TRANSFORMER_ID in markdown
    for target, labels in (("category", CATEGORIES), ("priority", PRIORITIES)):
        base = TargetMetrics.from_dict(baseline.metrics["test"][target])
        trans = TargetMetrics.from_dict(transformer.metrics["test"][target])
        for label in labels:
            b, t = base.per_class[label], trans.per_class[label]
            row = next(line for line in markdown.splitlines() if line.startswith(f"| {label} |"))
            for value in (b.precision, b.recall, b.f1, t.precision, t.recall, t.f1):
                assert f"{value:.4f}" in row
        for metrics, model in ((base, "Baseline"), (trans, "Transformer")):
            assert f"Confusion matrix ({target}, {model})" in markdown
            for matrix_row, label in zip(metrics.confusion_matrix, labels, strict=True):
                cells = " | ".join(str(cell) for cell in matrix_row)
                assert f"| {label} | {cells} |" in markdown
        assert f"{base.accuracy:.4f}" in markdown
        assert f"{trans.macro_f1:.4f}" in markdown


def test_model04_write_report_writes_markdown_and_json(tmp_path: Path) -> None:
    report = build_comparison(_baseline(), _transformer(0.65, 0.50))
    out_dir = tmp_path / "reports"

    md_path, json_path = write_report(report, out_dir)

    assert md_path == out_dir / "comparison.md"
    assert json_path == out_dir / "comparison.json"
    assert md_path.read_text(encoding="utf-8") == render_markdown(report)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS"
    assert data["macro_f1_delta_pp"] == {"category": 5.0, "priority": 0.0}
    assert data["baseline_id"] == BASELINE_ID
    assert data["transformer_id"] == TRANSFORMER_ID
    assert data["metrics"] == report.metrics


def test_model04_report_is_a_dataclass_with_contract_fields() -> None:
    report = ComparisonReport(
        baseline_id="b",
        transformer_id="t",
        metrics={},
        macro_f1_delta_pp={},
        verdict="FAIL",
    )

    assert report.verdict == "FAIL"


def _registry_with(tmp_path: Path, *versions: ModelVersion) -> ModelRegistry:
    registry = ModelRegistry(tmp_path)
    for version in versions:
        registry.register(version)
    return registry


def test_model04_cli_compare_without_transformer_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _registry_with(tmp_path, _baseline())

    exit_code = main(["compare", "--artifacts-dir", str(tmp_path)])

    assert exit_code == 1
    assert "No 'transformer' model version registered. Train it first." in capsys.readouterr().err
    assert not (tmp_path / "reports").exists()


def test_model04_cli_compare_without_baseline_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["compare", "--artifacts-dir", str(tmp_path)])

    assert exit_code == 1
    assert "No 'baseline' model version registered. Train it first." in capsys.readouterr().err


def test_model05_cli_compare_uses_latest_versions_and_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _registry_with(tmp_path, _baseline(), _transformer(0.60, 0.50))

    exit_code = main(["compare", "--artifacts-dir", str(tmp_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Verdict: FAIL" in out
    assert (tmp_path / "reports" / "comparison.md").is_file()
    assert (tmp_path / "reports" / "comparison.json").is_file()


def test_model05_cli_compare_uses_explicit_versions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other_transformer = _version(
        "transformer-20260103T000000Z-cccccccc",
        "transformer",
        0.10,
        0.10,
        created_at="2026-01-03T00:00:00+00:00",
    )
    _registry_with(tmp_path, _baseline(), _transformer(0.65, 0.50), other_transformer)

    exit_code = main(
        [
            "compare",
            "--artifacts-dir",
            str(tmp_path),
            "--baseline",
            BASELINE_ID,
            "--transformer",
            TRANSFORMER_ID,
        ]
    )

    assert exit_code == 0
    assert "Verdict: PASS" in capsys.readouterr().out
    data = json.loads((tmp_path / "reports" / "comparison.json").read_text(encoding="utf-8"))
    assert data["transformer_id"] == TRANSFORMER_ID


def test_model04_cli_compare_rejects_wrong_kind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _registry_with(tmp_path, _baseline(), _transformer(0.65, 0.50))

    exit_code = main(["compare", "--artifacts-dir", str(tmp_path), "--baseline", TRANSFORMER_ID])

    assert exit_code == 1
    assert capsys.readouterr().err.strip() != ""


def test_model04_cli_compare_different_splits_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _registry_with(tmp_path, _baseline(), _transformer(0.65, 0.50, splits_version="other"))

    exit_code = main(["compare", "--artifacts-dir", str(tmp_path)])

    assert exit_code == 1
    assert "Model versions were evaluated on different splits." in capsys.readouterr().err
