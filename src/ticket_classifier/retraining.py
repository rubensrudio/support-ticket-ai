"""Manual retraining with feedback and gated promotion (RETR-01..04, RETR-90..92).

Order: take the non-blocking ``artifacts/retrain.lock`` (DA-15) -> require a promoted
version -> require the current ``splits_version`` to match the promoted one (frozen
test set) -> load the current feedback -> train a new ``transformer`` version on
``train`` plus feedback -> promote it only if its test Macro F1 of ``category`` and
``priority`` are both >= the promoted ones (LAC-23).

Feedback is only ever appended to ``train``; ``validation`` and ``test`` are the
original splits. Any failure before ``promote`` leaves the promoted version unchanged.
"""

import fcntl
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ticket_classifier.data.splits import SPLIT_COLUMNS, splits_version
from ticket_classifier.errors import PipelineError
from ticket_classifier.labels import TARGETS
from ticket_classifier.pipeline_config import PipelineConfig
from ticket_classifier.registry import ModelRegistry, ModelVersion
from ticket_classifier.settings import Settings
from ticket_classifier.storage.database import connect, init_schema
from ticket_classifier.storage.feedback import FeedbackTrainingRow, list_current_feedback
from ticket_classifier.training import train_and_register

LOCK_FILE = "retrain.lock"

DECISION_PROMOTED: Literal["promoted"] = "promoted"
DECISION_NOT_PROMOTED: Literal["not promoted"] = "not promoted"

_NO_PROMOTED_MESSAGE = "No promoted model version. Train a transformer first."
_DATABASE_ERROR_MESSAGE = "Feedback database could not be accessed. Retraining aborted."


class RetrainInProgressError(PipelineError):
    """Another retraining run holds the retrain lock (RETR-92)."""

    def __init__(self) -> None:
        super().__init__("A retraining run is already in progress.")


class NoFeedbackError(PipelineError):
    """There is no current feedback to retrain with (RETR-90)."""

    def __init__(self) -> None:
        super().__init__("No feedback records available. Retraining aborted.")


@dataclass
class RetrainResult:
    """Outcome of a retraining run (RETR-04)."""

    feedback_rows: int
    new_version: ModelVersion
    previous_version: ModelVersion
    decision: Literal["promoted", "not promoted"]


@contextmanager
def _retrain_lock(settings: Settings) -> Iterator[None]:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.artifacts_dir / LOCK_FILE, "a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RetrainInProgressError() from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _check_frozen_test_set(current: str, promoted: ModelVersion) -> None:
    if current != promoted.splits_version:
        raise PipelineError(
            f"Frozen test set changed: splits version '{current}' differs from promoted "
            f"version '{promoted.splits_version}'. Retraining aborted."
        )


def _load_feedback(settings: Settings) -> list[FeedbackTrainingRow]:
    try:
        conn = connect(settings.db_path)
        try:
            init_schema(conn)
            return list_current_feedback(conn)
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        raise PipelineError(_DATABASE_ERROR_MESSAGE) from exc


def _feedback_frame(rows: list[FeedbackTrainingRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [(row.title, row.description, row.category, row.priority) for row in rows],
        columns=list(SPLIT_COLUMNS),
    )


def _passes_gate(new: ModelVersion, promoted: ModelVersion) -> bool:
    """LAC-23: every target's test Macro F1 must be >= the promoted one; missing fails."""
    for target in TARGETS:
        new_f1 = new.test_macro_f1(target)
        promoted_f1 = promoted.test_macro_f1(target)
        if new_f1 is None or promoted_f1 is None or new_f1 < promoted_f1:
            return False
    return True


def run_retraining(config: PipelineConfig, settings: Settings) -> RetrainResult:
    """Retrain a ``transformer`` version with the current feedback and gate its promotion.

    Raises ``RetrainInProgressError`` if another run holds the lock, ``PipelineError``
    if no version is promoted or the splits differ from the promoted version's, and
    ``NoFeedbackError`` if there is no feedback. Nothing is registered in those cases.
    """
    with _retrain_lock(settings):
        registry = ModelRegistry(settings.artifacts_dir)
        previous = registry.get_promoted()
        if previous is None:
            raise PipelineError(_NO_PROMOTED_MESSAGE)
        _check_frozen_test_set(splits_version(config.data.processed_dir), previous)

        rows = _load_feedback(settings)
        if not rows:
            raise NoFeedbackError()

        outcome = train_and_register(
            "transformer", config, settings, extra_train=_feedback_frame(rows)
        )
        new_version = outcome.version
        # The splits may have been rewritten while training: never gate on another test set.
        _check_frozen_test_set(new_version.splits_version, previous)

        decision: Literal["promoted", "not promoted"] = DECISION_NOT_PROMOTED
        if _passes_gate(new_version, previous):
            registry.promote(new_version.version_id)
            new_version = registry.get(new_version.version_id)
            decision = DECISION_PROMOTED
        return RetrainResult(
            feedback_rows=len(rows),
            new_version=new_version,
            previous_version=previous,
            decision=decision,
        )
