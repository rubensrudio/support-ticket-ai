"""Multi-head Transformer ticket classifier (DA-7, CT-13).

A shared pre-trained encoder feeds two linear heads applied to the first
token: one for ``category`` and one for ``priority``. Inference is serialized
by a lock so a single instance can be shared across request threads (DA-12).

``from_pretrained`` may download the encoder from the Hugging Face Hub (AS-8);
``load`` only reads local files from a saved artifact directory.
"""

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.base import ClassProbabilities

ENCODER_DIR = "encoder"
TOKENIZER_DIR = "tokenizer"
HEADS_FILE = "heads.pt"
META_FILE = "transformer_meta.json"
DROPOUT = 0.1
INFERENCE_BATCH_SIZE = 32


class MultiHeadModule(nn.Module):
    """Shared encoder with a category head and a priority head on the first token."""

    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(DROPOUT)
        self.category_head = nn.Linear(hidden_size, len(CATEGORIES))
        self.priority_head = nn.Linear(hidden_size, len(PRIORITIES))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(category_logits, priority_logits)`` for a batch."""
        inputs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**inputs).last_hidden_state[:, 0]
        hidden = self.dropout(hidden)
        return self.category_head(hidden), self.priority_head(hidden)

    def heads_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the state dict of both heads (encoder weights excluded)."""
        return {
            **{f"category_head.{k}": v for k, v in self.category_head.state_dict().items()},
            **{f"priority_head.{k}": v for k, v in self.priority_head.state_dict().items()},
        }

    def load_heads_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load weights produced by ``heads_state_dict``; strict on keys and shapes."""
        category = _sub_state(state, "category_head.")
        priority = _sub_state(state, "priority_head.")
        if len(category) + len(priority) != len(state):
            raise ValueError("heads state dict has unexpected keys")
        self.category_head.load_state_dict(category, strict=True)
        self.priority_head.load_state_dict(priority, strict=True)


def _sub_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def _validate_max_length(max_length: int) -> int:
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    return max_length


def _hidden_size(encoder: nn.Module) -> int:
    config: Any = getattr(encoder, "config", None)
    size = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
    if not isinstance(size, int) or size < 1:
        raise ValueError("encoder config does not expose a valid hidden size")
    return size


def _probabilities(logits: torch.Tensor, labels: Sequence[str]) -> list[dict[str, float]]:
    probs = torch.softmax(logits.float(), dim=-1).tolist()
    return [dict(zip(labels, (float(p) for p in row), strict=True)) for row in probs]


class TransformerClassifier:
    """Ticket classifier backed by a fine-tunable Transformer encoder (CT-13)."""

    kind: Literal["baseline", "transformer"] = "transformer"

    def __init__(
        self, module: MultiHeadModule, tokenizer: PreTrainedTokenizerBase, max_length: int
    ) -> None:
        self.module = module
        self.tokenizer = tokenizer
        self.max_length = _validate_max_length(max_length)
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, max_length: int, seed: int
    ) -> "TransformerClassifier":
        """Build a classifier from a pre-trained encoder with freshly seeded heads."""
        _validate_max_length(max_length)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        encoder = AutoModel.from_pretrained(model_name_or_path)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            module = MultiHeadModule(encoder, _hidden_size(encoder))
        return cls(module, tokenizer, max_length)

    @classmethod
    def load(cls, directory: Path) -> "TransformerClassifier":
        """Load a classifier saved by ``save``; reads only local files."""
        directory = Path(directory)
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"transformer artifact not found in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("invalid transformer metadata")
        if (
            tuple(meta.get("categories", ())) != CATEGORIES
            or tuple(meta.get("priorities", ())) != PRIORITIES
        ):
            raise ValueError("artifact labels do not match the configured label sets")
        max_length = _validate_max_length(meta.get("max_length"))  # type: ignore[arg-type]

        tokenizer = AutoTokenizer.from_pretrained(
            str(directory / TOKENIZER_DIR), local_files_only=True
        )
        encoder = AutoModel.from_pretrained(str(directory / ENCODER_DIR), local_files_only=True)
        module = MultiHeadModule(encoder, _hidden_size(encoder))
        state = torch.load(directory / HEADS_FILE, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("invalid heads file")
        module.load_heads_state_dict(state)
        module.eval()
        return cls(module, tokenizer, max_length)

    def save(self, directory: Path) -> None:
        """Write encoder, tokenizer, heads and metadata into ``directory`` (plan 7.4)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        encoder: Any = self.module.encoder
        with self._lock:
            encoder.save_pretrained(str(directory / ENCODER_DIR))
            self.tokenizer.save_pretrained(str(directory / TOKENIZER_DIR))
            torch.save(self.module.heads_state_dict(), directory / HEADS_FILE)
        meta = {
            "max_length": self.max_length,
            "categories": list(CATEGORIES),
            "priorities": list(PRIORITIES),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def encode(self, titles: Sequence[str], descriptions: Sequence[str]) -> BatchEncoding:
        """Tokenize (title, description) pairs, truncating to ``max_length`` tokens."""
        if len(titles) != len(descriptions):
            raise ValueError("titles and descriptions must have the same length")
        return self.tokenizer(
            list(titles),
            list(descriptions),
            truncation="longest_first",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

    def predict_proba(
        self, titles: Sequence[str], descriptions: Sequence[str]
    ) -> list[ClassProbabilities]:
        """Return category and priority probabilities for each ticket."""
        if len(titles) != len(descriptions):
            raise ValueError("titles and descriptions must have the same length")
        results: list[ClassProbabilities] = []
        with self._lock:
            self.module.eval()
            with torch.inference_mode():
                for start in range(0, len(titles), INFERENCE_BATCH_SIZE):
                    end = start + INFERENCE_BATCH_SIZE
                    encoding = self.encode(titles[start:end], descriptions[start:end])
                    category_logits, priority_logits = self.module(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        token_type_ids=encoding.get("token_type_ids"),
                    )
                    categories = _probabilities(category_logits, CATEGORIES)
                    priorities = _probabilities(priority_logits, PRIORITIES)
                    results.extend(
                        ClassProbabilities(category=c, priority=p)
                        for c, p in zip(categories, priorities, strict=True)
                    )
        return results
