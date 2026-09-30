"""Canonical label sets for the ticket classifier.

The tuple order is fixed: it is used in user-facing messages and to break ties.
This module is the single source of truth for label values.
"""

from typing import Literal

CATEGORIES: tuple[str, ...] = ("access", "infrastructure", "billing", "bug", "other")
PRIORITIES: tuple[str, ...] = ("low", "medium", "high")

Category = Literal["access", "infrastructure", "billing", "bug", "other"]
Priority = Literal["low", "medium", "high"]

TARGETS: tuple[str, str] = ("category", "priority")
