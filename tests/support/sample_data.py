"""Synthetic source dataset for data preparation tests (TASK-009).

The rows mimic the columns of the public source CSV (DA-3). Every canonical
category is reachable through ``configs/label_mapping.toml``. On top of
``per_category`` English rows per category, the fixture adds 10 non-English
rows, 5 exact duplicates, 3 rows with an empty ``subject`` and rows carrying
an e-mail address and a phone number.
"""

import csv
import json
import random
from pathlib import Path

from ticket_classifier.data.prepare import prepare_dataset
from ticket_classifier.pipeline_config import PipelineConfig, load_pipeline_config

SAMPLE_PER_CATEGORY = 160
NON_ENGLISH_ROWS = 10
DUPLICATE_ROWS = 5
EMPTY_SUBJECT_ROWS = 3

SAMPLE_EMAIL = "jane.doe@example.com"
SAMPLE_PHONE = "+1 555-123-4567"

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_MAPPING_PATH = REPO_ROOT / "configs" / "label_mapping.toml"

TAG_COLUMNS: tuple[str, ...] = tuple(f"tag_{index}" for index in range(1, 9))
SOURCE_COLUMNS: tuple[str, ...] = (
    "subject",
    "body",
    "answer",
    "type",
    "queue",
    "priority",
    "language",
    "version",
    *TAG_COLUMNS,
)

# category -> (source queue, first tag); resolved by configs/label_mapping.toml.
CATEGORY_SOURCES: dict[str, tuple[str, str]] = {
    "access": ("Technical Support", "Login"),
    "infrastructure": ("Service Outages and Maintenance", ""),
    "billing": ("Billing and Payments", ""),
    "bug": ("Product Support", "Bug"),
    "other": ("Customer Service", ""),
}
SOURCE_PRIORITIES: tuple[str, ...] = ("low", "medium", "high", "critical", "very_low")

_ADJECTIVES = ("urgent", "strange", "recurring", "minor", "new", "persistent", "odd")
_NOUNS = ("request", "problem", "question", "failure", "issue", "concern", "report")


def _row(
    subject: str,
    body: str,
    queue: str,
    priority: str,
    tag: str,
    language: str = "en",
) -> dict[str, str]:
    row = {column: "" for column in SOURCE_COLUMNS}
    row.update(
        subject=subject,
        body=body,
        answer="Thank you for reaching out.",
        type="Incident",
        queue=queue,
        priority=priority,
        language=language,
        version="1",
        tag_1=tag,
    )
    return row


def make_source_rows(per_category: int = SAMPLE_PER_CATEGORY, seed: int = 7) -> list[dict[str, str]]:
    """Build source CSV rows; see the module docstring for the composition."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for category, (queue, tag) in CATEGORY_SOURCES.items():
        for index in range(per_category):
            subject = f"{rng.choice(_ADJECTIVES)} {category} {rng.choice(_NOUNS)} {index}"
            body = f"Details about {category} case {index}.\\nPlease advise on next steps."
            if index % 40 == 0:
                body += f" Contact me at {SAMPLE_EMAIL} or {SAMPLE_PHONE}."
            rows.append(_row(subject, body, queue, rng.choice(SOURCE_PRIORITIES), tag))
    rng.shuffle(rows)

    queue, tag = CATEGORY_SOURCES["other"]
    for index in range(NON_ENGLISH_ROWS):
        rows.append(
            _row(f"Problem Nummer {index}", "Bitte helfen Sie mir.", queue, "low", tag, "de")
        )
    rows.extend(dict(rows[index]) for index in range(DUPLICATE_ROWS))
    for index in range(EMPTY_SUBJECT_ROWS):
        rows.append(_row("", f"Body without subject {index}.", queue, "medium", tag))
    return rows


def write_source_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write ``rows`` as a UTF-8 CSV with the source columns and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SOURCE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _toml_str(value: str) -> str:
    return json.dumps(value)


def write_pipeline_config(
    tmp_path: Path,
    source_path: Path,
    model_name: str = "unused",
    epochs: int = 1,
    max_length: int = 32,
    batch_size: int = 16,
) -> Path:
    """Write ``tmp_path/pipeline.toml`` with ``processed_dir`` under ``tmp_path``."""
    tag_columns = ", ".join(_toml_str(column) for column in TAG_COLUMNS)
    content = f"""seed = 42
[data]
source_path = {_toml_str(source_path.as_posix())}
processed_dir = {_toml_str((tmp_path / "processed").as_posix())}
title_column = "subject"
description_column = "body"
language_column = "language"
language = "en"
queue_column = "queue"
priority_column = "priority"
tag_columns = [{tag_columns}]
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15
min_train_examples_per_category = 100
[baseline]
c_grid = [1.0]
ngram_max = 2
min_df = 1
max_features = 5000
[transformer]
model_name = {_toml_str(model_name)}
max_length = {max_length}
epochs = {epochs}
batch_size = {batch_size}
learning_rate = 5e-5
weight_decay = 0.01
warmup_ratio = 0.1
num_threads = 1
"""
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def prepare_sample(tmp_path: Path, model_name: str = "unused") -> PipelineConfig:
    """Write the sample source and config under ``tmp_path``, prepare the splits
    and return the loaded config."""
    source_path = write_source_csv(tmp_path / "raw" / "tickets.csv", make_source_rows())
    config = load_pipeline_config(write_pipeline_config(tmp_path, source_path, model_name))
    prepare_dataset(config, LABEL_MAPPING_PATH)
    return config
