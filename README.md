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

## Training

Training needs the `train` extra (MLflow), installed by `uv sync --all-extras`. Train,
evaluate and register a model version from the prepared splits:

```bash
uv run ticket-classifier train-baseline
uv run ticket-classifier train-transformer
```

Both commands accept `--config` (default `configs/pipeline.toml`) and print
`Registered model version '<id>' (<kind>). Promoted: yes|no.` Each run:

- fails before training (exit code 1) if `train`, `validation` or `test` is empty or missing;
- trains on `train` only, selects hyperparameters on `validation` and never uses `test`
  for training;
- evaluates on `validation` and `test` and registers a new version with the metrics, the
  `splits_version`, the seed and the hyperparameters;
- promotes the version only if it is the first `transformer` version and no version is
  promoted yet.

Artifacts:

- Model files: `artifacts/models/<version_id>/` (the root is `TICKET_ARTIFACTS_DIR`,
  default `artifacts`).
- Registry: `artifacts/registry.json`. List the versions with
  `uv run ticket-classifier list-versions`.
- Experiment tracking: MLflow at `MLFLOW_TRACKING_URI` (default `sqlite:///mlflow.db`),
  with parameters, metrics, confusion matrices and the trained model.

> **Note:** the real Transformer training downloads `distilbert-base-uncased` from the
> Hugging Face Hub on first use. The test suite uses a local tiny model and never
> downloads anything.

## Retraining

Retraining is manual. It trains a new Transformer version on the original `train` split
plus the current feedback (the latest feedback of each stored prediction, with its masked
title and description), selects on the original `validation` split and evaluates on the
frozen `test` split, which never contains feedback.

Prerequisites:

- a promoted model version (`uv run ticket-classifier train-transformer`);
- the prepared splits used by the promoted version (same `splits_version`);
- at least one feedback record in the prediction database (`TICKET_DB_PATH`, default
  `var/tickets.db`).

```bash
uv run ticket-classifier retrain
```

The command accepts `--config` (default `configs/pipeline.toml`), prints the test Macro F1
of `category` and `priority` for the new and the promoted version, and ends with
`Retraining finished: <n> feedback records used. Decision: promoted.` or
`... Decision: not promoted.`

Promotion rule: the new version is promoted only if its test Macro F1 of `category` **and**
its test Macro F1 of `priority` are both greater than or equal to the promoted version's;
the previous version becomes `retired`. Otherwise the new version stays `registered` and
the promoted version is kept.

The command exits with code 1 and registers nothing if:

- another retraining run is in progress (`A retraining run is already in progress.`); runs
  are serialized with a lock on `artifacts/retrain.lock`;
- no version is promoted (`No promoted model version. Train a transformer first.`);
- the splits changed since the promoted version was trained
  (`Frozen test set changed: ... Retraining aborted.`);
- there is no feedback (`No feedback records available. Retraining aborted.`).

If training fails, the promoted version is kept and no version is promoted.

## Results

Compare the most recent Baseline and Transformer versions (or specific ones with
`--baseline ID` and `--transformer ID`) using the metrics recorded in the registry for
the `test` split. Models are not re-evaluated:

```bash
uv run ticket-classifier compare
```

The command writes `artifacts/reports/comparison.md` and `artifacts/reports/comparison.json`
(the root is `--artifacts-dir`, default `artifacts`) and prints a summary table with
`Verdict: PASS` or `Verdict: FAIL` (exit code 0 in both cases). The report shows, per
target (`category`, `priority`), accuracy, Macro F1, precision/recall/F1 per class and
the confusion matrix of both models, plus the Macro F1 difference in percentage points.

Verdict rule: `PASS` if the Transformer `category` Macro F1 is at least the Baseline one
plus 5 percentage points **and** the Transformer `priority` Macro F1 is at least the
Baseline one; `FAIL` otherwise.

The command exits with code 1 if no version of a kind is registered
(`No '<kind>' model version registered. Train it first.`) or if the two versions were
evaluated on different splits (`Model versions were evaluated on different splits.`).

## Run the API

Start the REST API locally (the app is built by a factory, hence `--factory`):

```bash
uv run uvicorn ticket_classifier.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

At startup the API creates the SQLite schema at `TICKET_DB_PATH` and loads the promoted
model version from the registry once. A version promoted while the API is running is
only served after a restart. If no promoted version can be loaded (no `registry.json`,
nothing promoted, or missing artifacts), the API still starts and reports itself as
unavailable. If `TICKET_API_KEY` is not set, the startup log contains
`API key not configured: /feedback will reject all requests.`

Health check (public, no API key required):

```bash
curl http://127.0.0.1:8000/health
```

- `200 {"status": "ok", "model_version": "<version_id>"}` when a model is loaded;
- `503 {"status": "unavailable", "model_version": null}` otherwise.

## Configuration

Runtime settings are read from environment variables:

| Variable                        | Default               | Description |
|---------------------------------|-----------------------|-------------|
| `TICKET_API_KEY`                | not set               | Key expected in the `X-API-Key` header of `POST /feedback`. Empty or unset means not configured: `/feedback` rejects every request. |
| `TICKET_REVIEW_THRESHOLD`       | `0.6`                 | Low-confidence threshold (`0.0` to `1.0`): a prediction whose category or priority confidence is below it gets `needs_review = true`. |
| `TICKET_DB_PATH`                | `var/tickets.db`      | SQLite database for predictions and feedback. The parent directory is created if needed. |
| `TICKET_ARTIFACTS_DIR`          | `artifacts`           | Root of the model artifacts and `registry.json`. |
| `TICKET_RETENTION_DAYS`         | `90`                  | Days a prediction without feedback is kept before it is purged (at least `1`). |
| `TICKET_PURGE_INTERVAL_SECONDS` | `86400`               | Interval between purge runs, in seconds (at least `1`). |
| `MLFLOW_TRACKING_URI`           | `sqlite:///mlflow.db` | MLflow tracking store, used by the training commands only (not by the API). |

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
