import logging
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def parse_crew_output(output, model: type[T], fallback: T) -> T:
    """Return a parsed Pydantic model from a CrewAI CrewOutput.

    Tries in order:
    1. output.pydantic (set when the crew task has output_pydantic configured)
    2. Parsing output.raw as JSON, stripping fenced code blocks if present
    3. Returns *fallback* and logs a warning
    """
    result: T | None = output.pydantic  # type: ignore[assignment]
    if result is not None:
        return result

    logger.warning("%s pydantic output missing — attempting raw parse", model.__name__)
    try:
        raw = output.raw or ""
        if "```" in raw:
            raw = raw.split("```")[-2].lstrip("json").strip()
        return model.model_validate_json(raw)
    except Exception as exc:
        logger.warning("Raw parse failed (%s) — using fallback %s", exc, model.__name__)
        return fallback
