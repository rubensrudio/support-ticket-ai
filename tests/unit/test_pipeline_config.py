from pathlib import Path

import pytest

from ticket_classifier.errors import PipelineError
from ticket_classifier.pipeline_config import (
    BaselineConfig,
    DataConfig,
    PipelineConfig,
    TransformerConfig,
    load_pipeline_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_TOML = REPO_ROOT / "configs" / "pipeline.toml"


def _valid_toml(
    *,
    ratios: tuple[str, str, str] = ("0.70", "0.15", "0.15"),
    include_transformer: bool = True,
    extra_data: str = "",
    drop_data_key: str | None = None,
) -> str:
    data_lines = [
        'source_path = "data/raw/tickets.csv"',
        'processed_dir = "data/processed"',
        'title_column = "subject"',
        'description_column = "body"',
        'language_column = "language"',
        'language = "en"',
        'queue_column = "queue"',
        'priority_column = "priority"',
        'tag_columns = ["tag_1", "tag_2"]',
        f"train_ratio = {ratios[0]}",
        f"validation_ratio = {ratios[1]}",
        f"test_ratio = {ratios[2]}",
        "min_train_examples_per_category = 100",
    ]
    if drop_data_key is not None:
        data_lines = [line for line in data_lines if not line.startswith(f"{drop_data_key} =")]
    parts = [
        "seed = 7",
        "[data]",
        *data_lines,
        extra_data,
        "[baseline]",
        "c_grid = [0.1, 1, 10.0]",
        "ngram_max = 2",
        "min_df = 2",
        "max_features = 50000",
    ]
    if include_transformer:
        parts += [
            "[transformer]",
            'model_name = "distilbert-base-uncased"',
            "max_length = 256",
            "epochs = 3",
            "batch_size = 16",
            "learning_rate = 5e-5",
            "weight_decay = 0.01",
            "warmup_ratio = 0.1",
            "num_threads = 4",
        ]
    return "\n".join(parts) + "\n"


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "pipeline.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_data04_loads_project_pipeline_toml() -> None:
    config = load_pipeline_config(PIPELINE_TOML)

    assert isinstance(config, PipelineConfig)
    assert config.seed == 42
    assert config.data.train_ratio == 0.70
    assert config.data.validation_ratio == 0.15
    assert config.data.test_ratio == 0.15
    assert config.data.min_train_examples_per_category == 100
    assert config.transformer.model_name == "distilbert-base-uncased"
    assert config.transformer.max_length == 256


def test_data04_project_config_field_types() -> None:
    config = load_pipeline_config(PIPELINE_TOML)

    assert isinstance(config.data, DataConfig)
    assert isinstance(config.baseline, BaselineConfig)
    assert isinstance(config.transformer, TransformerConfig)
    assert config.data.source_path == Path("data/raw/tickets.csv")
    assert config.data.processed_dir == Path("data/processed")
    assert config.data.tag_columns == tuple(f"tag_{i}" for i in range(1, 9))
    assert config.baseline.c_grid == (0.1, 1.0, 10.0)
    assert config.baseline.max_features == 50000
    assert config.transformer.learning_rate == 5e-5
    assert config.transformer.num_threads == 4


def test_data04_int_in_float_field_is_coerced(tmp_path: Path) -> None:
    config = load_pipeline_config(_write(tmp_path, _valid_toml()))

    assert config.baseline.c_grid == (0.1, 1.0, 10.0)
    assert all(isinstance(c, float) for c in config.baseline.c_grid)


def test_data04_config_is_frozen() -> None:
    config = load_pipeline_config(PIPELINE_TOML)

    with pytest.raises(AttributeError):
        config.seed = 1  # type: ignore[misc]


def test_data04_missing_transformer_section_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, _valid_toml(include_transformer=False))

    with pytest.raises(PipelineError) as exc_info:
        load_pipeline_config(path)

    assert "transformer" in exc_info.value.message


def test_data04_missing_key_is_named(tmp_path: Path) -> None:
    path = _write(tmp_path, _valid_toml(drop_data_key="test_ratio"))

    with pytest.raises(PipelineError) as exc_info:
        load_pipeline_config(path)

    assert "data.test_ratio" in exc_info.value.message


def test_data04_ratios_not_summing_to_one_raise(tmp_path: Path) -> None:
    path = _write(tmp_path, _valid_toml(ratios=("0.7", "0.2", "0.2")))

    with pytest.raises(PipelineError) as exc_info:
        load_pipeline_config(path)

    assert "ratio" in exc_info.value.message


def test_data04_wrong_type_raises(tmp_path: Path) -> None:
    content = _valid_toml().replace("max_length = 256", 'max_length = "256"')
    path = _write(tmp_path, content)

    with pytest.raises(PipelineError) as exc_info:
        load_pipeline_config(path)

    assert "transformer.max_length" in exc_info.value.message


def test_data04_bool_is_not_accepted_as_int(tmp_path: Path) -> None:
    content = _valid_toml().replace("seed = 7", "seed = true")
    path = _write(tmp_path, content)

    with pytest.raises(PipelineError) as exc_info:
        load_pipeline_config(path)

    assert "seed" in exc_info.value.message


def test_data04_tag_columns_must_be_strings(tmp_path: Path) -> None:
    content = _valid_toml().replace('tag_columns = ["tag_1", "tag_2"]', "tag_columns = [1, 2]")
    path = _write(tmp_path, content)

    with pytest.raises(PipelineError) as exc_info:
        load_pipeline_config(path)

    assert "data.tag_columns" in exc_info.value.message


def test_data04_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PipelineError):
        load_pipeline_config(tmp_path / "absent.toml")


def test_data04_invalid_toml_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "seed = = 42\n")

    with pytest.raises(PipelineError):
        load_pipeline_config(path)
