# Support Ticket AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

## License

Licensed under the **MIT License** — see [`LICENSE`](LICENSE).

Copyright © 2026 Rubens Rudio.
