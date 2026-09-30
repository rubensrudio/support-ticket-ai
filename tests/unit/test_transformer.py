import json
import os
import threading
from pathlib import Path

import pytest
import torch
from support.tiny_model import build_tiny_model

from ticket_classifier.labels import CATEGORIES, PRIORITIES
from ticket_classifier.models.base import ClassProbabilities, TicketClassifier
from ticket_classifier.models.transformer import TransformerClassifier

TITLES = ["cannot login", "server down", "invoice wrong", "app crashes on save"]
DESCRIPTIONS = [
    "password reset link does not work",
    "production host 10 is unreachable",
    "charged twice for plan 2",
    "error 500 when saving the form",
]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("tiny"))


@pytest.fixture()
def classifier(tiny_model_dir: Path) -> TransformerClassifier:
    return TransformerClassifier.from_pretrained(str(tiny_model_dir), max_length=32, seed=1)


def _as_tensors(results: list[ClassProbabilities]) -> tuple[torch.Tensor, torch.Tensor]:
    category = torch.tensor([[r.category[label] for label in CATEGORIES] for r in results])
    priority = torch.tensor([[r.priority[label] for label in PRIORITIES] for r in results])
    return category, priority


def test_model02_from_pretrained_works_offline(classifier: TransformerClassifier) -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert classifier.kind == "transformer"
    assert classifier.max_length == 32
    assert isinstance(classifier.module, torch.nn.Module)
    assert isinstance(classifier, TicketClassifier)


def test_model02_heads_have_expected_shape(classifier: TransformerClassifier) -> None:
    module = classifier.module
    assert module.category_head.out_features == len(CATEGORIES)
    assert module.priority_head.out_features == len(PRIORITIES)
    assert module.dropout.p == pytest.approx(0.1)


def test_model02_same_seed_gives_same_heads(tiny_model_dir: Path) -> None:
    first = TransformerClassifier.from_pretrained(str(tiny_model_dir), max_length=32, seed=7)
    second = TransformerClassifier.from_pretrained(str(tiny_model_dir), max_length=32, seed=7)

    assert torch.equal(first.module.category_head.weight, second.module.category_head.weight)
    assert torch.equal(first.module.priority_head.weight, second.module.priority_head.weight)


def test_model02_predict_proba_returns_all_labels_summing_to_one(
    classifier: TransformerClassifier,
) -> None:
    results = classifier.predict_proba(TITLES, DESCRIPTIONS)

    assert len(results) == len(TITLES)
    for result in results:
        assert isinstance(result, ClassProbabilities)
        assert set(result.category) == set(CATEGORIES)
        assert set(result.priority) == set(PRIORITIES)
        assert all(isinstance(v, float) for v in result.category.values())
        assert 0.999 <= sum(result.category.values()) <= 1.001
        assert 0.999 <= sum(result.priority.values()) <= 1.001


def test_model02_predict_proba_handles_more_than_one_batch(
    classifier: TransformerClassifier,
) -> None:
    titles = [f"title {i}" for i in range(70)]
    descriptions = [f"description {i}" for i in range(70)]

    batched = classifier.predict_proba(titles, descriptions)
    single = [
        classifier.predict_proba([t], [d])[0] for t, d in zip(titles, descriptions, strict=True)
    ]

    assert len(batched) == 70
    batched_cat, batched_pri = _as_tensors(batched)
    single_cat, single_pri = _as_tensors(single)
    assert torch.allclose(batched_cat, single_cat, atol=1e-5)
    assert torch.allclose(batched_pri, single_pri, atol=1e-5)


def test_model02_predict_proba_empty_input_returns_empty(
    classifier: TransformerClassifier,
) -> None:
    assert classifier.predict_proba([], []) == []


def test_model02_predict_proba_rejects_mismatched_lengths(
    classifier: TransformerClassifier,
) -> None:
    with pytest.raises(ValueError):
        classifier.predict_proba(["a", "b"], ["a"])


def test_model02_encode_uses_title_description_pair(classifier: TransformerClassifier) -> None:
    encoding = classifier.encode(["ab"], ["cd"])
    ids = encoding["input_ids"][0].tolist()
    sep_id = classifier.tokenizer.sep_token_id

    assert ids[0] == classifier.tokenizer.cls_token_id
    assert ids.count(sep_id) == 2


def test_api06_long_description_is_truncated_in_encode(
    classifier: TransformerClassifier,
) -> None:
    long_description = "x" * 5000

    encoding = classifier.encode(["cannot login"], [long_description])

    assert encoding["input_ids"].shape[-1] <= 32


def test_api06_long_description_does_not_break_predict_proba(
    classifier: TransformerClassifier,
) -> None:
    long_description = "error in module " * 400
    assert len(long_description) >= 5000

    results = classifier.predict_proba(["cannot login", "short"], [long_description, "ok"])

    assert len(results) == 2
    assert 0.999 <= sum(results[0].category.values()) <= 1.001


def test_model02_encode_pads_batch_to_same_length(classifier: TransformerClassifier) -> None:
    encoding = classifier.encode(["a", "a much longer title"], ["b", "and description"])

    assert encoding["input_ids"].shape[0] == 2
    assert encoding["attention_mask"][0].sum() < encoding["attention_mask"][1].sum()


def test_model02_save_and_load_reproduce_probabilities(
    classifier: TransformerClassifier, tmp_path: Path
) -> None:
    target = tmp_path / "artifact"
    classifier.save(target)

    assert (target / "encoder").is_dir()
    assert (target / "tokenizer").is_dir()
    assert (target / "heads.pt").is_file()
    meta = json.loads((target / "transformer_meta.json").read_text(encoding="utf-8"))
    assert meta == {
        "max_length": 32,
        "categories": list(CATEGORIES),
        "priorities": list(PRIORITIES),
    }

    loaded = TransformerClassifier.load(target)

    assert loaded.max_length == 32
    before_cat, before_pri = _as_tensors(classifier.predict_proba(TITLES, DESCRIPTIONS))
    after_cat, after_pri = _as_tensors(loaded.predict_proba(TITLES, DESCRIPTIONS))
    assert torch.allclose(before_cat, after_cat, atol=1e-6)
    assert torch.allclose(before_pri, after_pri, atol=1e-6)


def test_model02_load_rejects_label_mismatch(
    classifier: TransformerClassifier, tmp_path: Path
) -> None:
    target = tmp_path / "artifact"
    classifier.save(target)
    meta_path = target / "transformer_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["categories"] = ["access", "other"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError):
        TransformerClassifier.load(target)


def test_model02_load_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TransformerClassifier.load(tmp_path / "missing")


def test_model02_rejects_invalid_max_length(tiny_model_dir: Path) -> None:
    with pytest.raises(ValueError):
        TransformerClassifier.from_pretrained(str(tiny_model_dir), max_length=0, seed=1)


def test_model02_concurrent_predict_proba_matches_sequential(
    classifier: TransformerClassifier,
) -> None:
    expected = _as_tensors(classifier.predict_proba(TITLES, DESCRIPTIONS))
    outputs: list[list[ClassProbabilities] | None] = [None] * 8
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(index: int) -> None:
        try:
            barrier.wait()
            outputs[index] = classifier.predict_proba(TITLES, DESCRIPTIONS)
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    for output in outputs:
        assert output is not None
        category, priority = _as_tensors(output)
        assert torch.allclose(category, expected[0], atol=1e-6)
        assert torch.allclose(priority, expected[1], atol=1e-6)
