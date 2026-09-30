"""Dataset preparation: source CSV -> train/validation/test splits (DATA-01..DATA-11, DA-5).

Pipeline order (DA-5): read CSV -> keep ``language == config.data.language``
(``non_english``) -> decode the literal two-character ``\\n`` sequence into a
space -> ``preprocess_text`` on title/description -> drop empty text
(``empty_text``) -> map labels -> drop duplicate (title, description) keeping
the first (``duplicate``) -> stratified split -> check the per-category minimum
in ``train`` -> write. Every validation happens before anything is written, so
a failure never leaves partial splits behind (DATA-92, DATA-91, DATA-10).
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ticket_classifier.data.label_mapping import LabelMapping, load_label_mapping
from ticket_classifier.data.splits import SPLIT_NAMES, write_splits
from ticket_classifier.errors import PipelineError
from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.pipeline_config import DataConfig, PipelineConfig
from ticket_classifier.preprocessing import preprocess_text

_LITERAL_NEWLINE = "\\n"
_HASH_CHUNK_SIZE = 1 << 20
# Stratified splitting needs at least this many rows per category.
_MIN_ROWS_TO_STRATIFY = 3


class SourceDatasetNotFoundError(PipelineError):
    """The source dataset file does not exist or cannot be read (DATA-92)."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Source dataset not found at '{path}'. See README for download instructions."
        )
        self.path: Path = path


class InsufficientExamplesError(PipelineError):
    """A category has fewer training examples than the configured minimum (DATA-10)."""

    def __init__(self, category: str, count: int, minimum: int) -> None:
        super().__init__(
            f"Category '{category}' has {count} training examples; minimum is {minimum}."
        )
        self.category: str = category
        self.count: int = count
        self.minimum: int = minimum


def prepare_dataset(config: PipelineConfig, mapping_path: Path) -> dict[str, Any]:
    """Build the splits from ``config.data.source_path`` and write them to
    ``config.data.processed_dir``; return the preparation report (plan 7.2)."""
    data = config.data
    mapping = load_label_mapping(mapping_path)
    source, source_sha256 = _read_source(data)
    total_source_rows = len(source)

    english = source[source[data.language_column] == data.language]
    non_english = total_source_rows - len(english)

    records = _preprocess(english, data)
    non_empty = records[(records["title"] != "") & (records["description"] != "")]
    empty_text = len(records) - len(non_empty)

    labelled = _map_labels(non_empty, english.loc[non_empty.index], data, mapping)
    unique = labelled.drop_duplicates(subset=["title", "description"], keep="first")
    duplicate = len(labelled) - len(unique)

    splits = _split(unique.reset_index(drop=True), config)
    _check_min_train_examples(splits["train"], data.min_train_examples_per_category)

    report: dict[str, Any] = {
        "source_path": str(data.source_path),
        "source_sha256": source_sha256,
        "seed": config.seed,
        "total_source_rows": total_source_rows,
        "discarded": {
            "non_english": non_english,
            "empty_text": empty_text,
            "duplicate": duplicate,
        },
        "splits": {name: _split_summary(splits[name]) for name in SPLIT_NAMES},
    }
    version = write_splits(splits, report, data.processed_dir)
    return {**report, "splits_version": version}


def _read_source(data: DataConfig) -> tuple[pd.DataFrame, str]:
    path = data.source_path
    if not path.is_file():
        raise SourceDatasetNotFoundError(path)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
        # dtype=str + keep_default_na=False: empty cells stay "" instead of NaN,
        # so no null value can reach preprocessing or the written splits.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        raise SourceDatasetNotFoundError(path) from None

    required = [
        data.title_column,
        data.description_column,
        data.language_column,
        data.queue_column,
        data.priority_column,
        *data.tag_columns,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PipelineError(
            f"Source dataset '{path}' is missing required columns: {', '.join(missing)}."
        )
    return frame, digest.hexdigest()


def _clean_text(value: str) -> str:
    return preprocess_text(value.replace(_LITERAL_NEWLINE, " "))


def _preprocess(frame: pd.DataFrame, data: DataConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [_clean_text(value) for value in frame[data.title_column]],
            "description": [_clean_text(value) for value in frame[data.description_column]],
        },
        index=frame.index,
        dtype=object,
    )


def _map_labels(
    records: pd.DataFrame, source: pd.DataFrame, data: DataConfig, mapping: LabelMapping
) -> pd.DataFrame:
    tags = source.loc[:, list(data.tag_columns)].itertuples(index=False, name=None)
    categories = [
        mapping.map_category(queue, row_tags)
        for queue, row_tags in zip(source[data.queue_column], tags, strict=True)
    ]
    priorities = [mapping.map_priority(value) for value in source[data.priority_column]]
    labelled = records.copy()
    labelled["category"] = pd.Series(categories, index=records.index, dtype=object)
    labelled["priority"] = pd.Series(priorities, index=records.index, dtype=object)
    return labelled


def _split(frame: pd.DataFrame, config: PipelineConfig) -> dict[str, pd.DataFrame]:
    data = config.data
    counts = frame["category"].value_counts()
    for category in CATEGORIES:
        available = int(counts.get(category, 0))
        if available < _MIN_ROWS_TO_STRATIFY:
            # Too few rows to stratify: the train split can hold at most ``available``.
            raise InsufficientExamplesError(
                category, available, data.min_train_examples_per_category
            )

    labels = frame["category"].to_numpy()
    indices = np.arange(len(frame))
    validation_share = data.validation_ratio / (1.0 - data.test_ratio)
    try:
        rest, test = train_test_split(
            indices, test_size=data.test_ratio, stratify=labels, random_state=config.seed
        )
        train, validation = train_test_split(
            rest, test_size=validation_share, stratify=labels[rest], random_state=config.seed
        )
    except ValueError as exc:
        raise PipelineError(
            f"Dataset could not be split with the configured ratios: {exc}"
        ) from None

    parts = {"train": train, "validation": validation, "test": test}
    return {name: frame.iloc[parts[name]].reset_index(drop=True) for name in SPLIT_NAMES}


def _check_min_train_examples(train: pd.DataFrame, minimum: int) -> None:
    counts = train["category"].value_counts()
    for category in CATEGORIES:
        count = int(counts.get(category, 0))
        if count < minimum:
            raise InsufficientExamplesError(category, count, minimum)


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    categories = frame["category"].value_counts()
    priorities = frame["priority"].value_counts()
    return {
        "total": len(frame),
        "category": {label: int(categories.get(label, 0)) for label in CATEGORIES},
        "priority": {label: int(priorities.get(label, 0)) for label in PRIORITIES},
    }
