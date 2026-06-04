from __future__ import annotations

import logging
import re

from src.agents.models.aggregation import AggregatedValue
from src.agents.models.analysis import DataPoint
from src.agents.models.intent import QueryIntent

logger = logging.getLogger(__name__)

_THRESHOLD_RE = re.compile(
    r"\b(over|above|more than|greater than|under|below|less than)\s+[\£\$\€]?([\d,]+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def _parse_threshold(query: str) -> tuple[str, float] | None:
    """Return (operator, value) if query contains a numeric threshold."""
    m = _THRESHOLD_RE.search(query)
    if not m:
        return None
    word = m.group(1).lower()
    val = float(m.group(2).replace(",", ""))
    op = ">" if word in ("over", "above", "more than", "greater than") else "<"
    return op, val


def aggregate(
    datapoints: list[DataPoint],
    intent: QueryIntent,
    query: str,
) -> AggregatedValue | None:
    """
    Compute a numeric aggregate over pertinent DataPoints when structured intent
    is present. Returns None if no aggregation is applicable.
    """
    if "structured" not in intent:
        return None

    pertinent = [dp for dp in datapoints if dp.pertinent and dp.value is not None]
    if not pertinent:
        logger.info("aggregate: no pertinent DataPoints with a value — skipping")
        return None

    values = [dp.value for dp in pertinent]  # type: ignore[misc]
    ids = [dp.uri for dp in pertinent]
    q = query.lower()

    # Threshold filter
    threshold = _parse_threshold(query)
    if threshold:
        op, cutoff = threshold
        filtered = [(dp, v) for dp, v in zip(pertinent, values)
                    if (v > cutoff if op == ">" else v < cutoff)]
        result_val = float(len(filtered))
        logger.info("aggregate: filter %s %.2f → %d matching", op, cutoff, int(result_val))
        return AggregatedValue(
            operation="filter",
            result=result_val,
            unit="matching items",
            contributing_point_ids=[dp.uri for dp, _ in filtered],
        )

    # Max / min
    if any(w in q for w in ["most", "highest", "maximum", "largest", "biggest"]):
        result_val = max(values)
        op_name = "max"
    elif any(w in q for w in ["least", "lowest", "minimum", "smallest", "fewest"]):
        result_val = min(values)
        op_name = "min"
    # Average
    elif any(w in q for w in ["average", "mean", "avg"]):
        result_val = sum(values) / len(values)
        op_name = "mean"
    # Sum (default for "total", "how many", "count")
    else:
        result_val = sum(values)
        op_name = "sum"

    logger.info("aggregate: %s of %d values = %.2f", op_name, len(values), result_val)
    return AggregatedValue(
        operation=op_name,
        result=result_val,
        unit=None,
        contributing_point_ids=ids,
    )
