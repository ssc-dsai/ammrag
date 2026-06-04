from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AggregatedValue:
    operation: str          # "sum", "max", "min", "mean", "filter"
    result: float
    unit: str | None
    contributing_point_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{self.operation}={self.result}{unit_str} (from {len(self.contributing_point_ids)} source(s))"
