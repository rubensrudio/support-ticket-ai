import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from ticket_classifier.labels import CATEGORIES, PRIORITIES, TARGETS
from ticket_classifier.models import baseline as baseline_module
from ticket_classifier.models.base import ClassProbabilities, TicketClassifier
from ticket_classifier.models.baseline import BaselineClassifier, train_baseline
from ticket_classifier.pipeline_config import BaselineConfig

KEYWORDS = {
    "access": ["login", "password", "account", "locked"],
    "infrastructure": ["server", "network", "outage", "latency"],
    "billing": ["invoice", "charge", "refund", "payment"],
    "bug": ["crash", "error", "exception", "broken"],
    "other": ["question", "feedback", "suggestion", "general"],
}

CONFIG = BaselineConfig(c_grid=(0.1, 1.0, 10.0), ngram_max=2, min_df=1, max_features=5000)

BATCH_TITLES = ["cannot login", "server outage", "invoice charge", "app crash", "a question"]
BATCH_DESCRIPTIONS = [
    "password account locked",
    "network latency high",
    "refund payment missing",
    "error exception broken",
    "general feedback suggestion",
]


def _frame(rows_per_category: int, offset: int) -> pd.DataFrame:
    rows = []
    for category, words in KEYWORDS.items():
        for i in range(rows_per_category):
            n = i + offset
            rows.append(
                {
                    "title": f"{words[n % 4]} {words[(n + 1) % 4]}",
                    "description": f"{words[(n + 2) % 4]} issue number {n}",
                    "category": category,
                    "priority": PRIORITIES[n % len(PRIORITIES)],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _frame(20, 0), _frame(6, 100)


@pytest.fixture(scope="module")
def trained(
    splits: tuple[pd.DataFrame, pd.DataFrame],
) -> tuple[BaselineClassifier, dict[str, object]]:
    train, validation = splits
    return train_baseline(train, validation, CONFIG, seed=42)


def test_model01_returns_chosen_c_per_target(
    trained: tuple[BaselineClassifier, dict[str, object]],
) -> None:
    classifier, params = trained
    assert classifier.kind == "baseline"
    assert isinstance(classifier, TicketClassifier)
    assert set(params) == set(TARGETS)
    for target in TARGETS:
        target_params = params[target]
        assert isinstance(target_params, dict)
        assert target_params["C"] in CONFIG.c_grid
        assert target_params["ngram_max"] == CONFIG.ngram_max
        assert target_params["min_df"] == CONFIG.min_df
        assert target_params["max_features"] == CONFIG.max_features
        assert target_params["class_weight"] == "balanced"


def test_model01_predict_proba_has_all_labels_and_sums_to_one(
    trained: tuple[BaselineClassifier, dict[str, object]],
) -> None:
    classifier, _ = trained
    results = classifier.predict_proba(BATCH_TITLES, BATCH_DESCRIPTIONS)

    assert len(results) == len(BATCH_TITLES)
    for result in results:
        assert isinstance(result, ClassProbabilities)
        assert tuple(result.category) == CATEGORIES
        assert tuple(result.priority) == PRIORITIES
        assert 0.999 <= sum(result.category.values()) <= 1.001
        assert 0.999 <= sum(result.priority.values()) <= 1.001
        assert all(isinstance(v, float) for v in result.category.values())


def test_model01_learns_keyword_categories(
    trained: tuple[BaselineClassifier, dict[str, object]],
) -> None:
    classifier, _ = trained
    results = classifier.predict_proba(BATCH_TITLES, BATCH_DESCRIPTIONS)
    predicted = [max(r.category, key=r.category.__getitem__) for r in results]
    assert predicted == list(CATEGORIES)


def test_model01_same_seed_gives_identical_probabilities(
    splits: tuple[pd.DataFrame, pd.DataFrame],
    trained: tuple[BaselineClassifier, dict[str, object]],
) -> None:
    train, validation = splits
    first, first_params = trained
    second, second_params = train_baseline(train, validation, CONFIG, seed=42)

    assert first_params == second_params
    assert first.predict_proba(BATCH_TITLES, BATCH_DESCRIPTIONS) == second.predict_proba(
        BATCH_TITLES, BATCH_DESCRIPTIONS
    )


def test_model01_save_and_load_reproduce_probabilities(
    tmp_path: Path, trained: tuple[BaselineClassifier, dict[str, object]]
) -> None:
    classifier, params = trained
    classifier.save(tmp_path)

    assert (tmp_path / "category.joblib").is_file()
    assert (tmp_path / "priority.joblib").is_file()
    meta = json.loads((tmp_path / "baseline_meta.json").read_text(encoding="utf-8"))
    assert meta["categories"] == list(CATEGORIES)
    assert meta["priorities"] == list(PRIORITIES)
    assert meta["hyperparameters"] == params

    loaded = BaselineClassifier.load(tmp_path)
    assert loaded.kind == "baseline"
    assert loaded.predict_proba(BATCH_TITLES, BATCH_DESCRIPTIONS) == classifier.predict_proba(
        BATCH_TITLES, BATCH_DESCRIPTIONS
    )


def test_model01_load_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BaselineClassifier.load(tmp_path)


def test_model01_load_rejects_mismatched_labels(
    tmp_path: Path, trained: tuple[BaselineClassifier, dict[str, object]]
) -> None:
    classifier, _ = trained
    classifier.save(tmp_path)
    meta_path = tmp_path / "baseline_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["categories"] = ["access", "billing"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError):
        BaselineClassifier.load(tmp_path)


def test_model07_signature_does_not_accept_test_split() -> None:
    parameters = list(inspect.signature(train_baseline).parameters)
    assert parameters == ["train", "validation", "config", "seed"]
    assert "test" not in parameters


def test_model01_tie_on_validation_picks_first_c_of_grid(
    monkeypatch: pytest.MonkeyPatch, splits: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, validation = splits
    monkeypatch.setattr(baseline_module, "f1_score", lambda *args, **kwargs: 0.5)
    config = BaselineConfig(c_grid=(10.0, 0.1, 1.0), ngram_max=1, min_df=1, max_features=100)
    _, params = train_baseline(train, validation, config, seed=0)
    for target in TARGETS:
        assert params[target]["C"] == 10.0


def test_model01_predict_proba_rejects_mismatched_lengths(
    trained: tuple[BaselineClassifier, dict[str, object]],
) -> None:
    classifier, _ = trained
    with pytest.raises(ValueError):
        classifier.predict_proba(["one"], [])


def test_model01_rejects_missing_columns(splits: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, validation = splits
    with pytest.raises(ValueError):
        train_baseline(train.drop(columns=["priority"]), validation, CONFIG, seed=0)


def test_model01_rejects_unknown_labels(splits: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, validation = splits
    bad = train.copy()
    bad.loc[0, "category"] = "unknown"
    with pytest.raises(ValueError):
        train_baseline(bad, validation, CONFIG, seed=0)


def test_model01_rejects_empty_grid(splits: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, validation = splits
    config = BaselineConfig(c_grid=(), ngram_max=1, min_df=1, max_features=100)
    with pytest.raises(ValueError):
        train_baseline(train, validation, config, seed=0)
