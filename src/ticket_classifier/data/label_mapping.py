"""Versioned mapping from source dataset labels to canonical labels (DATA-02, DATA-91).

Format (decision LAC-32 = A, DA-4):

- ``[priority]``: every source priority -> canonical priority.
- ``[category.queue]``: every source queue -> default canonical category.
- ``[[category.rules]]``: ordered rules; the first match wins. A rule matches by
  ``queue_in`` (source queue) or ``any_tag_in`` (any non-empty source tag).
  Tags not referenced by any rule are ignored.
"""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ticket_classifier.errors import PipelineError
from ticket_classifier.labels import CATEGORIES, PRIORITIES


class UnmappedLabelError(PipelineError):
    """A source label is missing from the label mapping table (DATA-91)."""

    def __init__(self, label: str) -> None:
        super().__init__(f"Unmapped source label '{label}'. Add it to the label mapping table.")
        self.label: str = label


@dataclass(frozen=True)
class CategoryRule:
    """One ordered category rule; exactly one of ``queue_in``/``any_tag_in`` is set."""

    name: str
    category: str
    queue_in: frozenset[str] = frozenset()
    any_tag_in: frozenset[str] = frozenset()

    def matches(self, queue: str, tags: Sequence[str]) -> bool:
        if self.queue_in:
            return queue in self.queue_in
        return any(tag.strip() in self.any_tag_in for tag in tags if tag and tag.strip())


@dataclass(frozen=True)
class LabelMapping:
    """Loaded label mapping table. Build it with :func:`load_label_mapping`."""

    version: str
    priority: Mapping[str, str]
    category_queue: Mapping[str, str]
    category_rules: tuple[CategoryRule, ...]

    def map_category(self, queue: str, tags: Sequence[str]) -> str:
        """Return the canonical category; raise ``UnmappedLabelError`` for unknown queues."""
        if queue not in self.category_queue:
            raise UnmappedLabelError(queue)
        for rule in self.category_rules:
            if rule.matches(queue, tags):
                return rule.category
        return self.category_queue[queue]

    def map_priority(self, source: str) -> str:
        """Return the canonical priority; raise ``UnmappedLabelError`` for unknown values."""
        try:
            return self.priority[source]
        except KeyError:
            raise UnmappedLabelError(source) from None


def load_label_mapping(path: Path) -> LabelMapping:
    """Load and validate the mapping table; raise ``PipelineError`` if it is invalid."""
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        raise PipelineError(f"Label mapping table not found: {path}") from None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PipelineError(f"Label mapping table {path} could not be read: {exc}") from None

    version = raw.get("version", "")
    if not isinstance(version, str):
        raise PipelineError("Label mapping table: 'version' must be a string.")

    priority = _string_table(raw.get("priority"), "priority", PRIORITIES)
    category = raw.get("category")
    if not isinstance(category, dict):
        raise PipelineError("Label mapping table: missing [category] table.")
    category_queue = _string_table(category.get("queue"), "category.queue", CATEGORIES)
    rules = _rules(category.get("rules", []))

    return LabelMapping(
        version=version,
        priority=MappingProxyType(priority),
        category_queue=MappingProxyType(category_queue),
        category_rules=rules,
    )


def _string_table(value: object, name: str, allowed: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise PipelineError(f"Label mapping table: missing or empty [{name}] table.")
    table: dict[str, str] = {}
    for source, target in value.items():
        if not isinstance(target, str) or target not in allowed:
            raise PipelineError(
                f"Label mapping table: [{name}] maps '{source}' to invalid label "
                f"'{target}'. Allowed: {', '.join(allowed)}."
            )
        table[source] = target
    return table


def _rules(value: object) -> tuple[CategoryRule, ...]:
    if not isinstance(value, list):
        raise PipelineError("Label mapping table: [[category.rules]] must be an array of tables.")
    rules: list[CategoryRule] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise PipelineError(f"Label mapping table: category rule #{index} must be a table.")
        name = item.get("name", f"#{index}")
        label = f"category rule '{name}'"
        target = item.get("category")
        if not isinstance(target, str) or target not in CATEGORIES:
            raise PipelineError(
                f"Label mapping table: {label} has invalid category '{target}'. "
                f"Allowed: {', '.join(CATEGORIES)}."
            )
        has_queue = "queue_in" in item
        has_tags = "any_tag_in" in item
        if has_queue == has_tags:
            raise PipelineError(
                f"Label mapping table: {label} must define exactly one of "
                "'queue_in' or 'any_tag_in'."
            )
        key = "queue_in" if has_queue else "any_tag_in"
        values = _string_set(item[key], f"{label} '{key}'")
        rules.append(
            CategoryRule(
                name=str(name),
                category=target,
                queue_in=values if has_queue else frozenset(),
                any_tag_in=values if has_tags else frozenset(),
            )
        )
    return tuple(rules)


def _string_set(value: object, name: str) -> frozenset[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(entry, str) and entry.strip() for entry in value)
    ):
        raise PipelineError(f"Label mapping table: {name} must be a non-empty list of strings.")
    return frozenset(entry.strip() for entry in value)
