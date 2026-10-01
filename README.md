# ticket-classifier

## Overview

`ticket-classifier` is a support ticket classification project: a training pipeline
(baseline and transformer models) and a REST API that serves predictions. The package
exposes a single command-line entry point, `ticket-classifier`.

## Requirements

- Python 3.12 (`>=3.12,<3.13`)
- [uv](https://docs.astral.sh/uv/) for dependency management and builds
- No GPU required: `torch` is installed from the CPU-only PyTorch index

## Setup

Install all dependencies (runtime, the `train` extra with MLflow, and the `dev` group):

```bash
uv sync --all-extras
```

Check the CLI:

```bash
uv run ticket-classifier --help
```

## Dataset

The models are trained on a public dataset that is **not** committed to this repository
and is never downloaded by code. Fetch it manually.

| Field    | Value                                                              |
|----------|--------------------------------------------------------------------|
| Source   | Hugging Face dataset `Tobi-Bueck/customer-support-tickets`         |
| File     | `dataset-tickets-multi-lang-4-20k.csv`                             |
| Revision | `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`                         |
| SHA-256  | `9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635` |
| License  | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only) |

> **Note on data origin:** the dataset author also publishes a synthetic ticket generator
> and states the data contains "no PII"; the origin of the tickets is not declared, so the
> dataset may be machine-generated and metrics on it may not reflect real tickets. PII
> masking still applies to all text, including real tickets sent to the API.

Download the pinned revision and verify its checksum:

```bash
mkdir -p data/raw
curl -L -o data/raw/tickets.csv \
  https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb/dataset-tickets-multi-lang-4-20k.csv
echo "9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635  data/raw/tickets.csv" | sha256sum -c -
```

Prepare the train/validation/test splits:

```bash
uv run ticket-classifier prepare
```

Options: `--config` (default `configs/pipeline.toml`) and `--mapping` (default
`configs/label_mapping.toml`). The command keeps only English rows (`language` column),
masks PII, drops empty and duplicate tickets, maps source labels to canonical labels and
writes a stratified 70/15/15 split with a fixed seed to `data/processed/`:
`train.jsonl`, `validation.jsonl`, `test.jsonl` and `prepare_report.json` (source SHA-256,
discarded rows by reason, label counts per split). If the source file is missing, a
source label is unmapped, or a category has fewer than 100 training examples, the command
exits with code 1 and writes nothing.

## Development

| Task      | Command                                              |
|-----------|------------------------------------------------------|
| Lint      | `uv run ruff check . --output-format=concise`        |
| Typecheck | `uv run mypy src`                                    |
| Test      | `uv run pytest --junitxml=reports/junit.xml`         |
| Build     | `uv build --wheel --out-dir dist`                    |

The test run writes a JUnit report to `reports/junit.xml`. Tests marked `container`
(which need a running Docker container) are excluded by default.

## Tests run offline

The test suite runs without a GPU and without internet access. `tests/conftest.py` sets
the following environment variables before any test module is imported:

- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`: Hugging Face libraries never try to
  download models or datasets.
- `CUDA_VISIBLE_DEVICES=""`: PyTorch never sees a GPU, so everything runs on CPU.

Tests must use local fixtures only; any test that would need network access fails
instead of silently downloading data.
