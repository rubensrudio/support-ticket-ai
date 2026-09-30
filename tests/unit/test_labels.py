import typing

from ticket_classifier.labels import CATEGORIES, PRIORITIES, TARGETS, Category, Priority


def test_data02_categories_have_fixed_order() -> None:
    assert CATEGORIES == ("access", "infrastructure", "billing", "bug", "other")


def test_data02_priorities_have_fixed_order() -> None:
    assert PRIORITIES == ("low", "medium", "high")


def test_data02_category_literal_matches_categories() -> None:
    assert typing.get_args(Category) == CATEGORIES


def test_data02_priority_literal_matches_priorities() -> None:
    assert typing.get_args(Priority) == PRIORITIES


def test_data02_targets_are_category_and_priority() -> None:
    assert TARGETS == ("category", "priority")
