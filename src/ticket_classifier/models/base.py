"""Shared classifier contract and probability helpers.

Concrete classifiers (baseline and transformer) implement ``TicketClassifier``
and return one ``ClassProbabilities`` per ticket.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class ClassProbabilities:
    """Per-label probabilities for one ticket.

    Each mapping has every label of its target as a key and sums to about 1.
    """

    category: dict[str, float]
    priority: dict[str, float]


@runtime_checkable
class TicketClassifier(Protocol):
    """Structural interface shared by every ticket classifier."""

    kind: Literal["baseline", "transformer"]

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        """Return one ``ClassProbabilities`` per (title, description) pair."""
        ...

    def save(self, directory: Path) -> None:
        """Persist the classifier artifacts into ``directory``."""
        ...


def top_label(probs: Mapping[str, float], order: Sequence[str]) -> tuple[str, float]:
    """Return the most probable label and its exact probability.

    Ties are broken by the position of the label in ``order`` (earliest wins).
    Raises ``ValueError`` when ``probs`` is empty or has a label not in ``order``.
    """
    unknown = set(probs) - set(order)
    if unknown:
        raise ValueError(f"labels not in order: {sorted(unknown)}")
    best: tuple[str, float] | None = None
    for label in order:
        if label not in probs:
            continue
        value = probs[label]
        if best is None or value > best[1]:
            best = (label, value)
    if best is None:
        raise ValueError("probabilities are empty")
    return best
