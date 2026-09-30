"""TF-IDF + Logistic Regression baseline classifier (CT-12, DA-16).

One scikit-learn ``Pipeline`` is trained per target (``category`` and
``priority``) on the ``train`` split only; the regularization strength ``C`` is
chosen by Macro F1 on the ``validation`` split. The ``test`` split is never
seen here (MODEL-07).

Artifacts are written with ``joblib`` (pickle): only load directories produced
locally by this pipeline (plan 7.4).
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from ticket_classifier.labels import CATEGORIES, PRIORITIES, TARGETS
from ticket_classifier.models.base import ClassProbabilities
from ticket_classifier.pipeline_config import BaselineConfig

META_FILE = "baseline_meta.json"
MAX_ITER = 2000
CLASS_WEIGHT = "balanced"
TEXT_COLUMNS: tuple[str, str] = ("title", "description")

_LABELS: dict[str, tuple[str, ...]] = {"category": CATEGORIES, "priority": PRIORITIES}


def _artifact_file(target: str) -> str:
    return f"{target}.joblib"


def _join_text(titles: Sequence[str], descriptions: Sequence[str]) -> list[str]:
    if len(titles) != len(descriptions):
        raise ValueError("titles and descriptions must have the same length")
    return [
        f"{title} {description}" for title, description in zip(titles, descriptions, strict=True)
    ]


def _frame_text(frame: pd.DataFrame) -> list[str]:
    return _join_text(
        frame["title"].astype(str).tolist(), frame["description"].astype(str).tolist()
    )


def _validate_frame(frame: pd.DataFrame, name: str) -> None:
    required = (*TEXT_COLUMNS, *TARGETS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} split is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} split is empty")
    for target in TARGETS:
        unknown = set(frame[target].astype(str)) - set(_LABELS[target])
        if unknown:
            raise ValueError(f"{name} split has unknown {target} labels: {sorted(unknown)}")


def _validate_config(config: BaselineConfig) -> None:
    if not config.c_grid:
        raise ValueError("baseline c_grid must not be empty")
    if any(c <= 0 for c in config.c_grid):
        raise ValueError("baseline c_grid values must be positive")
    if config.ngram_max < 1 or config.min_df < 1 or config.max_features < 1:
        raise ValueError("baseline ngram_max, min_df and max_features must be positive")


def _build_pipeline(config: BaselineConfig, c: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, config.ngram_max),
                    min_df=config.min_df,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight=CLASS_WEIGHT,
                    max_iter=MAX_ITER,
                    random_state=seed,
                ),
            ),
        ]
    )


def _probabilities(
    pipeline: Pipeline, texts: list[str], labels: Sequence[str]
) -> list[dict[str, float]]:
    classes = [str(c) for c in pipeline.classes_]
    rows = pipeline.predict_proba(texts)
    results: list[dict[str, float]] = []
    for row in rows:
        by_class = dict(zip(classes, (float(p) for p in row), strict=True))
        # Labels absent from the train split get probability 0.0 so every key exists.
        results.append({label: by_class.get(label, 0.0) for label in labels})
    return results


class BaselineClassifier:
    """Ticket classifier with one TF-IDF + Logistic Regression pipeline per target."""

    kind: Literal["baseline", "transformer"] = "baseline"

    def __init__(
        self,
        category: Pipeline,
        priority: Pipeline,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        self.pipelines: dict[str, Pipeline] = {"category": category, "priority": priority}
        self.hyperparameters: dict[str, Any] = dict(hyperparameters or {})

    @classmethod
    def load(cls, directory: Path) -> "BaselineClassifier":
        """Load a classifier saved by ``save``; only use locally produced artifacts."""
        directory = Path(directory)
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"baseline artifact not found in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("invalid baseline metadata")
        if (
            tuple(meta.get("categories", ())) != CATEGORIES
            or tuple(meta.get("priorities", ())) != PRIORITIES
        ):
            raise ValueError("artifact labels do not match the configured label sets")
        pipelines: dict[str, Pipeline] = {}
        for target in TARGETS:
            path = directory / _artifact_file(target)
            if not path.is_file():
                raise FileNotFoundError(f"baseline artifact not found in {directory}")
            pipeline = joblib.load(path)
            if not isinstance(pipeline, Pipeline):
                raise ValueError(f"invalid baseline artifact for {target}")
            pipelines[target] = pipeline
        hyperparameters = meta.get("hyperparameters", {})
        if not isinstance(hyperparameters, dict):
            raise ValueError("invalid baseline metadata")
        return cls(pipelines["category"], pipelines["priority"], hyperparameters)

    def save(self, directory: Path) -> None:
        """Write ``category.joblib``, ``priority.joblib`` and metadata (plan 7.4)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for target in TARGETS:
            joblib.dump(self.pipelines[target], directory / _artifact_file(target))
        meta = {
            "categories": list(CATEGORIES),
            "priorities": list(PRIORITIES),
            "hyperparameters": self.hyperparameters,
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        """Return category and priority probabilities for each ticket."""
        texts = _join_text(titles, descriptions)
        if not texts:
            return []
        categories = _probabilities(self.pipelines["category"], texts, CATEGORIES)
        priorities = _probabilities(self.pipelines["priority"], texts, PRIORITIES)
        return [
            ClassProbabilities(category=c, priority=p)
            for c, p in zip(categories, priorities, strict=True)
        ]


def train_baseline(
    train: pd.DataFrame, validation: pd.DataFrame, config: BaselineConfig, seed: int
) -> tuple[BaselineClassifier, dict[str, Any]]:
    """Train one pipeline per target on ``train`` and pick ``C`` by validation Macro F1.

    Ties keep the first ``C`` of ``config.c_grid``. The winning pipeline is the
    one fitted on ``train`` only (no refit on train + validation). Returns the
    classifier and the chosen hyperparameters per target.
    """
    _validate_config(config)
    _validate_frame(train, "train")
    _validate_frame(validation, "validation")

    train_text = _frame_text(train)
    validation_text = _frame_text(validation)

    chosen: dict[str, Pipeline] = {}
    hyperparameters: dict[str, Any] = {}
    for target in TARGETS:
        y_train = train[target].astype(str).tolist()
        y_validation = validation[target].astype(str).tolist()
        best: tuple[float, float, Pipeline] | None = None
        for c in config.c_grid:
            pipeline = _build_pipeline(config, c, seed)
            pipeline.fit(train_text, y_train)
            predicted = pipeline.predict(validation_text)
            score = float(f1_score(y_validation, predicted, average="macro", zero_division=0))
            if best is None or score > best[1]:
                best = (c, score, pipeline)
        assert best is not None  # c_grid is validated as non-empty
        chosen[target] = best[2]
        hyperparameters[target] = {
            "C": best[0],
            "ngram_max": config.ngram_max,
            "min_df": config.min_df,
            "max_features": config.max_features,
            "class_weight": CLASS_WEIGHT,
            "max_iter": MAX_ITER,
        }

    classifier = BaselineClassifier(chosen["category"], chosen["priority"], hyperparameters)
    return classifier, hyperparameters
