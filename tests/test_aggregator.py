"""Unit tests for numeric aggregation — no live services required."""
import pytest

from src.agents.aggregator import aggregate
from src.agents.models.analysis import DataPoint


def make_point(value: float, uri: str = "test://x") -> DataPoint:
    return DataPoint(uri=uri, type="text", pertinent=True,
                     step="test", reasoning="test", quote="test", value=value)


def make_non_pertinent(value: float) -> DataPoint:
    return DataPoint(uri="test://np", type="text", pertinent=False,
                     step="test", reasoning="not relevant", quote="", value=value)


POINTS = [
    make_point(120.0, "uri-a"),
    make_point(95.0,  "uri-b"),
    make_point(60.0,  "uri-c"),
]


def test_sum():
    result = aggregate(POINTS, ["structured"], "total employees across all departments")
    assert result is not None
    assert result.operation == "sum"
    assert result.result == 275.0
    assert len(result.contributing_point_ids) == 3


def test_max():
    result = aggregate(POINTS, ["structured"], "which office has the most employees")
    assert result is not None
    assert result.operation == "max"
    assert result.result == 120.0


def test_min():
    result = aggregate(POINTS, ["structured"], "which office has the fewest employees")
    assert result is not None
    assert result.operation == "min"
    assert result.result == 60.0


def test_mean():
    result = aggregate(POINTS, ["structured"], "average headcount across offices")
    assert result is not None
    assert result.operation == "mean"
    assert abs(result.result - 91.666) < 0.01


def test_threshold_filter():
    invoice_points = [
        make_point(5000.0,  "inv-1"),
        make_point(12000.0, "inv-2"),
        make_point(15000.0, "inv-3"),
        make_point(800.0,   "inv-4"),
    ]
    result = aggregate(invoice_points, ["structured"], "invoices over £10,000")
    assert result is not None
    assert result.operation == "filter"
    assert result.result == 2.0  # inv-2 and inv-3
    assert "inv-2" in result.contributing_point_ids
    assert "inv-3" in result.contributing_point_ids


def test_non_pertinent_excluded():
    points = POINTS + [make_non_pertinent(9999.0)]
    result = aggregate(points, ["structured"], "total employees")
    assert result is not None
    assert result.result == 275.0  # 9999 excluded


def test_no_structured_intent_returns_none():
    result = aggregate(POINTS, ["navigational"], "what is WBA")
    assert result is None


def test_no_values_returns_none():
    points = [DataPoint(uri="u", type="text", pertinent=True,
                        step="s", reasoning="r", quote="q", value=None)]
    result = aggregate(points, ["structured"], "total")
    assert result is None
