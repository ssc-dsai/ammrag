from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VALID_TAGS = {"spatial", "visual", "structured", "navigational", "temporal", "compound"}

QueryIntent = list[str]


@dataclass
class TemporalConstraint:
    after: str | None = None   # ISO 8601 date
    before: str | None = None  # ISO 8601 date
    anchor: str | None = None  # e.g. "2022-01-01" for "since 2022"
