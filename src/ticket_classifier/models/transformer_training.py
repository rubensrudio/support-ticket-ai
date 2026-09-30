"""Deterministic fine-tuning loop for the Transformer classifier (CT-14).

Only the ``train`` split is used for gradient updates and only the
``validation`` split is used to choose the best epoch; the ``test`` split is
never received here (MODEL-07). Determinism follows DA-8 and class imbalance
is handled with inverse-frequency weighted cross-entropy (DA-16).
"""

import random
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from ticket_classifier.evaluation.metrics import evaluate_classifier
from ticket_classifier.labels import CATEGORIES, PRIORITIES, TARGETS
from ticket_classifier.models.transformer import TransformerClassifier
from ticket_classifier.pipeline_config import TransformerConfig

TEXT_COLUMNS: tuple[str, str] = ("title", "description")

_LABELS: dict[str, tuple[str, ...]] = {"category": CATEGORIES, "priority": PRIORITIES}


def set_determinism(seed: int, num_threads: int) -> None:
    """Seed every random source and force deterministic torch kernels (DA-8)."""
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


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


def _validate_config(config: TransformerConfig) -> None:
    if config.epochs < 1:
        raise ValueError("transformer epochs must be positive")
    if config.batch_size < 1:
        raise ValueError("transformer batch_size must be positive")
    if config.max_length < 1:
        raise ValueError("transformer max_length must be positive")
    if config.learning_rate <= 0:
        raise ValueError("transformer learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("transformer weight_decay must not be negative")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("transformer warmup_ratio must be in [0, 1)")
    if config.num_threads < 1:
        raise ValueError("transformer num_threads must be positive")


def _label_indices(values: pd.Series, labels: Sequence[str]) -> torch.Tensor:
    index = {label: position for position, label in enumerate(labels)}
    return torch.tensor([index[str(value)] for value in values], dtype=torch.long)


def _inverse_frequency_weights(targets: torch.Tensor, num_labels: int) -> torch.Tensor:
    """Weight each label by ``n / (k * count)``; labels absent from train get 0."""
    counts = torch.bincount(targets, minlength=num_labels).float()
    present = counts > 0
    weights = torch.zeros(num_labels, dtype=torch.float)
    weights[present] = targets.numel() / (int(present.sum()) * counts[present])
    return weights


def _mean_macro_f1(classifier: TransformerClassifier, frame: pd.DataFrame) -> float:
    metrics = evaluate_classifier(classifier, frame)
    return sum(metrics[target].macro_f1 for target in TARGETS) / len(TARGETS)


def train_transformer(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: TransformerConfig,
    seed: int,
) -> tuple[TransformerClassifier, dict[str, Any]]:
    """Fine-tune the Transformer on ``train`` and keep the best ``validation`` epoch.

    Returns the classifier loaded with the best checkpoint (highest mean of
    category and priority Macro F1 on ``validation``) and a dict with the
    hyperparameters, the seed and ``best_epoch`` (1-based).
    """
    _validate_frame(train, "train")
    _validate_frame(validation, "validation")
    _validate_config(config)

    set_determinism(seed, config.num_threads)
    classifier = TransformerClassifier.from_pretrained(config.model_name, config.max_length, seed)
    module = classifier.module

    encoding = classifier.encode(
        [str(value) for value in train["title"]],
        [str(value) for value in train["description"]],
    )
    category_targets = _label_indices(train["category"], CATEGORIES)
    priority_targets = _label_indices(train["priority"], PRIORITIES)
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(encoding["input_ids"])
        use_token_types = False
    else:
        use_token_types = True
    dataset = TensorDataset(
        encoding["input_ids"],
        encoding["attention_mask"],
        token_type_ids,
        category_targets,
        priority_targets,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    category_loss = nn.CrossEntropyLoss(
        weight=_inverse_frequency_weights(category_targets, len(CATEGORIES))
    )
    priority_loss = nn.CrossEntropyLoss(
        weight=_inverse_frequency_weights(priority_targets, len(PRIORITIES))
    )
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = config.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        module.train()
        for input_ids, attention_mask, types, category, priority in loader:
            optimizer.zero_grad()
            category_logits, priority_logits = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=types if use_token_types else None,
            )
            loss = category_loss(category_logits, category) + priority_loss(
                priority_logits, priority
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

        score = _mean_macro_f1(classifier, validation)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in module.state_dict().items()}

    assert best_state is not None  # epochs is validated as positive
    module.load_state_dict(best_state)
    module.eval()

    info: dict[str, Any] = {
        **asdict(config),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
    }
    return classifier, info
