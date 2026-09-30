"""Experiment tracking of training runs in MLflow (OPS-04).

Training-only module: it needs the optional ``train`` extra (``mlflow``) and must not be
imported by the API, services, storage, registry or model loader.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ticket_classifier.errors import PipelineError
from ticket_classifier.registry import ModelVersion

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "support-ticket-classifier"
MODEL_ARTIFACT_PATH = "model"
PER_CLASS_METRICS = ("precision", "recall", "f1")

_METRIC_NAME_UNSAFE = re.compile(r"[^\w\-. /]")


def tracking_uri() -> str:
    """Tracking URI from ``MLFLOW_TRACKING_URI``, falling back to a local SQLite store."""
    return os.environ.get(TRACKING_URI_ENV) or DEFAULT_TRACKING_URI


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested hyperparameters into dotted keys with string values."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, name))
        else:
            flat[name] = str(value)
    return flat


def run_params(version: ModelVersion) -> dict[str, str]:
    """All parameters logged for a training run."""
    params = flatten_params(version.hyperparameters)
    params.update(
        {
            "seed": str(version.seed),
            "splits_version": version.splits_version,
            "kind": version.kind,
            "version_id": version.version_id,
            "feedback_rows": str(version.feedback_rows),
        }
    )
    return params


def run_metrics(version: ModelVersion) -> dict[str, float]:
    """Accuracy, macro F1 and per-label precision/recall/F1 for every split and target."""
    metrics: dict[str, float] = {}
    for split, targets in version.metrics.items():
        for target, values in targets.items():
            base = f"{split}_{target}"
            for name in ("accuracy", "macro_f1"):
                if _is_number(values.get(name)):
                    metrics[_metric_name(f"{base}_{name}")] = float(values[name])
            per_class = values.get("per_class") or {}
            for label, class_values in per_class.items():
                for name in PER_CLASS_METRICS:
                    value = class_values.get(name) if isinstance(class_values, Mapping) else None
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        metrics[_metric_name(f"{base}_{label}_{name}")] = float(value)
    return metrics


def confusion_matrices(version: ModelVersion) -> dict[str, dict[str, Any]]:
    """Confusion matrix artifacts keyed by ``confusion_matrix_{split}_{target}.json``."""
    artifacts: dict[str, dict[str, Any]] = {}
    for split, targets in version.metrics.items():
        for target, values in targets.items():
            matrix = values.get("confusion_matrix")
            if matrix is None:
                continue
            artifacts[f"confusion_matrix_{split}_{target}.json"] = {
                "labels": list(values.get("labels") or []),
                "matrix": matrix,
            }
    return artifacts


def log_training_run(
    version: ModelVersion,
    model_dir: Path,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> str:
    """Log one training run (params, metrics, confusion matrices, model) and return its run id."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise PipelineError(f"Model directory not found for version {version.version_id}.")

    try:
        import mlflow
        from mlflow.exceptions import MlflowException
    except ImportError as exc:
        raise PipelineError(
            "Experiment tracking requires the 'train' extra (mlflow) to be installed."
        ) from exc

    try:
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=version.version_id) as run:
            mlflow.log_params(run_params(version))
            mlflow.log_metrics(run_metrics(version))
            for file_name, payload in confusion_matrices(version).items():
                mlflow.log_dict(payload, file_name)
            mlflow.log_artifacts(str(model_dir), artifact_path=MODEL_ARTIFACT_PATH)
            run_id: str = run.info.run_id
    except (MlflowException, OSError) as exc:
        raise PipelineError(
            f"Failed to log training run for version {version.version_id} to MLflow."
        ) from exc
    return run_id


def _metric_name(name: str) -> str:
    return _METRIC_NAME_UNSAFE.sub("_", name)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
