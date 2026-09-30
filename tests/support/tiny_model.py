"""Offline tiny BERT fixture for transformer tests (DA-9).

Builds a randomly initialised ``BertModel`` and a matching tokenizer from a
local vocabulary, so tests never download anything from the Hugging Face Hub.
"""

import string
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
_SYMBOLS: tuple[str, ...] = tuple(string.ascii_lowercase + string.digits)
_PUNCTUATION: tuple[str, ...] = tuple(".,:;!?'\"()-_/@#&%+=")


def _vocabulary() -> list[str]:
    return [
        *SPECIAL_TOKENS,
        *_SYMBOLS,
        *(f"##{symbol}" for symbol in _SYMBOLS),
        *_PUNCTUATION,
    ]


def build_tiny_model(directory: Path) -> Path:
    """Write a tiny BERT encoder and tokenizer into ``directory`` and return it.

    The result can be passed to ``AutoModel.from_pretrained`` and
    ``AutoTokenizer.from_pretrained`` with ``HF_HUB_OFFLINE=1``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    vocab = _vocabulary()
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast.from_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    model = BertModel(config)
    model.save_pretrained(str(directory))
    return directory
