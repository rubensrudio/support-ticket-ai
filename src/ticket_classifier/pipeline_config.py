"""Typed loader for ``configs/pipeline.toml`` (training pipeline parameters)."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ticket_classifier.errors import PipelineError

_RATIO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DataConfig:
    source_path: Path
    processed_dir: Path
    title_column: str
    description_column: str
    language_column: str
    language: str
    queue_column: str
    priority_column: str
    tag_columns: tuple[str, ...]
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    min_train_examples_per_category: int


@dataclass(frozen=True)
class BaselineConfig:
    c_grid: tuple[float, ...]
    ngram_max: int
    min_df: int
    max_features: int


@dataclass(frozen=True)
class TransformerConfig:
    model_name: str
    max_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_threads: int


@dataclass(frozen=True)
class PipelineConfig:
    seed: int
    data: DataConfig
    baseline: BaselineConfig
    transformer: TransformerConfig


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Read and validate the pipeline TOML file.

    Raises ``PipelineError`` naming the offending key when the file is missing,
    malformed, lacks a key, has a value of the wrong type, or when the split
    ratios do not sum to 1.0.
    """
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        raise PipelineError(f"Pipeline config file not found: {path}") from None
    except OSError:
        raise PipelineError(f"Pipeline config file could not be read: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise PipelineError(f"Pipeline config file is not valid TOML: {path} ({exc})") from None

    data = _parse_data(_section(raw, "data"))
    return PipelineConfig(
        seed=_int(raw, "seed", ""),
        data=data,
        baseline=_parse_baseline(_section(raw, "baseline")),
        transformer=_parse_transformer(_section(raw, "transformer")),
    )


def _parse_data(table: dict[str, Any]) -> DataConfig:
    prefix = "data."
    config = DataConfig(
        source_path=Path(_str(table, "source_path", prefix)),
        processed_dir=Path(_str(table, "processed_dir", prefix)),
        title_column=_str(table, "title_column", prefix),
        description_column=_str(table, "description_column", prefix),
        language_column=_str(table, "language_column", prefix),
        language=_str(table, "language", prefix),
        queue_column=_str(table, "queue_column", prefix),
        priority_column=_str(table, "priority_column", prefix),
        tag_columns=_str_tuple(table, "tag_columns", prefix),
        train_ratio=_float(table, "train_ratio", prefix),
        validation_ratio=_float(table, "validation_ratio", prefix),
        test_ratio=_float(table, "test_ratio", prefix),
        min_train_examples_per_category=_int(table, "min_train_examples_per_category", prefix),
    )
    total = config.train_ratio + config.validation_ratio + config.test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_RATIO_TOLERANCE):
        raise PipelineError(
            "Pipeline config keys data.train_ratio, data.validation_ratio and "
            f"data.test_ratio must sum to 1.0 (got {total})"
        )
    return config


def _parse_baseline(table: dict[str, Any]) -> BaselineConfig:
    prefix = "baseline."
    return BaselineConfig(
        c_grid=_float_tuple(table, "c_grid", prefix),
        ngram_max=_int(table, "ngram_max", prefix),
        min_df=_int(table, "min_df", prefix),
        max_features=_int(table, "max_features", prefix),
    )


def _parse_transformer(table: dict[str, Any]) -> TransformerConfig:
    prefix = "transformer."
    return TransformerConfig(
        model_name=_str(table, "model_name", prefix),
        max_length=_int(table, "max_length", prefix),
        epochs=_int(table, "epochs", prefix),
        batch_size=_int(table, "batch_size", prefix),
        learning_rate=_float(table, "learning_rate", prefix),
        weight_decay=_float(table, "weight_decay", prefix),
        warmup_ratio=_float(table, "warmup_ratio", prefix),
        num_threads=_int(table, "num_threads", prefix),
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = _require(raw, name, "")
    if not isinstance(value, dict):
        raise PipelineError(f"Pipeline config key '{name}' must be a table")
    return value


def _require(table: dict[str, Any], key: str, prefix: str) -> Any:
    if key not in table:
        raise PipelineError(f"Pipeline config is missing key '{prefix}{key}'")
    return table[key]


def _type_error(prefix: str, key: str, expected: str) -> PipelineError:
    return PipelineError(f"Pipeline config key '{prefix}{key}' must be {expected}")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _str(table: dict[str, Any], key: str, prefix: str) -> str:
    value = _require(table, key, prefix)
    if not isinstance(value, str):
        raise _type_error(prefix, key, "a string")
    return value


def _int(table: dict[str, Any], key: str, prefix: str) -> int:
    value = _require(table, key, prefix)
    if not isinstance(value, int) or isinstance(value, bool):
        raise _type_error(prefix, key, "an integer")
    return value


def _float(table: dict[str, Any], key: str, prefix: str) -> float:
    value = _require(table, key, prefix)
    if not _is_number(value):
        raise _type_error(prefix, key, "a number")
    return float(value)


def _str_tuple(table: dict[str, Any], key: str, prefix: str) -> tuple[str, ...]:
    value = _require(table, key, prefix)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _type_error(prefix, key, "an array of strings")
    return tuple(value)


def _float_tuple(table: dict[str, Any], key: str, prefix: str) -> tuple[float, ...]:
    value = _require(table, key, prefix)
    if not isinstance(value, list) or not all(_is_number(item) for item in value):
        raise _type_error(prefix, key, "an array of numbers")
    return tuple(float(item) for item in value)
