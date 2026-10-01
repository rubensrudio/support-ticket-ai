"""Train, evaluate, track and register a model version (MODEL-06..09, MODEL-90, OPS-04).

Order: load the three splits (fails before training if any is empty or missing) ->
append ``extra_train`` to ``train`` -> train (never with ``test``) -> evaluate on
``validation`` and ``test`` -> save artifacts -> log the MLflow run -> register ->
promote the first ``transformer`` version when nothing is promoted yet.
"""

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd

from ticket_classifier.data.splits import SPLIT_COLUMNS, SPLIT_NAMES, load_split, splits_version
from ticket_classifier.errors import PipelineError
from ticket_classifier.evaluation.metrics import evaluate_classifier
from ticket_classifier.models.base import TicketClassifier
from ticket_classifier.models.baseline import train_baseline
from ticket_classifier.models.transformer_training import train_transformer
from ticket_classifier.pipeline_config import PipelineConfig
from ticket_classifier.registry import STATUS_REGISTERED, ModelRegistry, ModelVersion
from ticket_classifier.settings import Settings
from ticket_classifier.tracking import log_training_run

EVALUATION_SPLITS: tuple[str, str] = ("validation", "test")


@dataclass
class TrainingOutcome:
    """Result of one training run: the registered version and whether it was promoted."""

    version: ModelVersion
    promoted: bool


def _combined_train(train: pd.DataFrame, extra_train: pd.DataFrame | None) -> pd.DataFrame:
    if extra_train is None:
        return train
    missing = [column for column in SPLIT_COLUMNS if column not in extra_train.columns]
    if missing:
        raise PipelineError(f"Extra training data is missing columns: {', '.join(missing)}.")
    extra = extra_train.loc[:, list(SPLIT_COLUMNS)].astype(str)
    return pd.concat([train, extra], ignore_index=True)


def _train(
    kind: str, train: pd.DataFrame, validation: pd.DataFrame, config: PipelineConfig
) -> tuple[TicketClassifier, dict[str, Any]]:
    if kind == "baseline":
        return train_baseline(train, validation, config.baseline, config.seed)
    return train_transformer(train, validation, config.transformer, config.seed)


def train_and_register(
    kind: Literal["baseline", "transformer"],
    config: PipelineConfig,
    settings: Settings,
    extra_train: pd.DataFrame | None = None,
) -> TrainingOutcome:
    """Train a ``kind`` model on the prepared splits and register it as a new version.

    The ``test`` split is only used for evaluation after training. The version is
    promoted only if ``kind == "transformer"``, ``extra_train is None`` and no version
    is promoted yet (MODEL-09).
    """
    if kind not in ("baseline", "transformer"):
        raise PipelineError("Unknown model kind; expected one of: baseline, transformer.")

    data_dir = config.data.processed_dir
    splits = {name: load_split(data_dir, name) for name in SPLIT_NAMES}
    version_of_splits = splits_version(data_dir)
    train = _combined_train(splits["train"], extra_train)

    classifier, hyperparameters = _train(kind, train, splits["validation"], config)
    metrics = {
        split: {
            target: values.to_dict()
            for target, values in evaluate_classifier(classifier, splits[split]).items()
        }
        for split in EVALUATION_SPLITS
    }

    registry = ModelRegistry(settings.artifacts_dir)
    version_id = registry.new_version_id(kind)
    model_dir = registry.model_dir(version_id)
    version = ModelVersion(
        version_id=version_id,
        kind=kind,
        status=STATUS_REGISTERED,
        created_at=datetime.now(UTC).isoformat(),
        splits_version=version_of_splits,
        seed=config.seed,
        hyperparameters=hyperparameters,
        metrics=metrics,
        training_rows=len(train),
        artifact_dir=model_dir.relative_to(registry.artifacts_dir).as_posix(),
        feedback_rows=0 if extra_train is None else len(extra_train),
    )
    try:
        classifier.save(model_dir)
        log_training_run(version, model_dir)
        registry.register(version)
    except BaseException:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise

    promoted = False
    if kind == "transformer" and extra_train is None and registry.get_promoted() is None:
        registry.promote(version_id)
        promoted = True
        version = registry.get(version_id)
    return TrainingOutcome(version=version, promoted=promoted)
