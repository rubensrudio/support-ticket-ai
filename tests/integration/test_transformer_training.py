import inspect
from pathlib import Path

import pandas as pd
import pytest
import torch
from support.tiny_model import build_tiny_model

from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.transformer import TransformerClassifier
from ticket_classifier.models.transformer_training import set_determinism, train_transformer
from ticket_classifier.pipeline_config import TransformerConfig

ROWS_PER_CATEGORY = 20
SEED = 7

_KEYWORDS = {
    "access": ("login", "password"),
    "infrastructure": ("server", "network"),
    "billing": ("invoice", "charge"),
    "bug": ("crash", "error"),
    "other": ("question", "feedback"),
}


def _frame(rows_per_category: int, offset: int = 0) -> pd.DataFrame:
    records = []
    for category in CATEGORIES:
        first, second = _KEYWORDS[category]
        for index in range(rows_per_category):
            number = index + offset
            records.append(
                {
                    "title": f"{first} issue {number}",
                    "description": f"the {second} is not ok for user {number}",
                    "category": category,
                    "priority": PRIORITIES[number % len(PRIORITIES)],
                }
            )
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("tiny_training"))


@pytest.fixture(scope="module")
def config(tiny_model_dir: Path) -> TransformerConfig:
    return TransformerConfig(
        model_name=str(tiny_model_dir),
        max_length=32,
        epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_threads=1,
    )


@pytest.fixture(scope="module")
def train_frame() -> pd.DataFrame:
    return _frame(ROWS_PER_CATEGORY)


@pytest.fixture(scope="module")
def validation_frame() -> pd.DataFrame:
    return _frame(4, offset=100)


def _probabilities(classifier: TransformerClassifier, frame: pd.DataFrame) -> torch.Tensor:
    results = classifier.predict_proba(frame["title"].tolist(), frame["description"].tolist())
    return torch.tensor(
        [
            [r.category[label] for label in CATEGORIES]
            + [r.priority[label] for label in PRIORITIES]
            for r in results
        ]
    )


def test_model02_train_transformer_returns_best_epoch_and_hyperparameters(
    config: TransformerConfig, train_frame: pd.DataFrame, validation_frame: pd.DataFrame
) -> None:
    classifier, info = train_transformer(train_frame, validation_frame, config, SEED)

    assert isinstance(classifier, TransformerClassifier)
    assert info["best_epoch"] in {1, 2}
    assert info["model_name"] == config.model_name
    assert info["max_length"] == 32
    assert info["epochs"] == 2
    assert info["batch_size"] == 16
    assert info["learning_rate"] == config.learning_rate
    assert info["weight_decay"] == config.weight_decay
    assert info["warmup_ratio"] == config.warmup_ratio
    assert info["num_threads"] == 1
    assert info["seed"] == SEED
    assert not classifier.module.training


def test_model06_same_seed_gives_same_probabilities(
    config: TransformerConfig, train_frame: pd.DataFrame, validation_frame: pd.DataFrame
) -> None:
    first, first_info = train_transformer(train_frame, validation_frame, config, SEED)
    second, second_info = train_transformer(train_frame, validation_frame, config, SEED)

    batch = validation_frame.head(8)
    assert first_info == second_info
    assert torch.allclose(_probabilities(first, batch), _probabilities(second, batch), atol=1e-6)


def test_model07_train_transformer_signature_has_no_test_split() -> None:
    parameters = list(inspect.signature(train_transformer).parameters)

    assert parameters == ["train", "validation", "config", "seed"]
    assert "test" not in parameters


def test_model02_rejects_frame_with_unknown_label(
    config: TransformerConfig, train_frame: pd.DataFrame, validation_frame: pd.DataFrame
) -> None:
    broken = train_frame.copy()
    broken.loc[0, "category"] = "unknown"

    with pytest.raises(ValueError, match="train"):
        train_transformer(broken, validation_frame, config, SEED)


def test_model02_rejects_invalid_config(
    config: TransformerConfig, train_frame: pd.DataFrame, validation_frame: pd.DataFrame
) -> None:
    invalid = TransformerConfig(**{**config.__dict__, "epochs": 0})

    with pytest.raises(ValueError, match="epochs"):
        train_transformer(train_frame, validation_frame, invalid, SEED)


def test_model06_set_determinism_seeds_torch_and_threads() -> None:
    set_determinism(3, 1)
    first = torch.rand(4)
    set_determinism(3, 1)
    second = torch.rand(4)

    assert torch.equal(first, second)
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()


def test_model06_set_determinism_rejects_invalid_threads() -> None:
    with pytest.raises(ValueError, match="num_threads"):
        set_determinism(3, 0)
