from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator

_NONE_STRINGS = {"none", "null", "n/a", "na", "", "-"}


def _coerce_none_string(v: Any) -> Any:
    """Convert LLM 'none'/'N/A'/'' strings to actual None."""
    if isinstance(v, str) and v.strip().lower() in _NONE_STRINGS:
        return None
    return v


class DataPoint(BaseModel):
    uri: str = Field(description="URI of the source file")
    type: Literal["image", "text"] = Field(description="Whether this data point is an image or text")
    pertinent: bool = Field(description="Whether this vector is pertinent to the step")
    step: str = Field(description="The plan step (info field) this was assessed against")
    reasoning: str = Field(description="Why the vector is or is not pertinent to the step")
    quote: str = Field(default="", description="Verbatim excerpt from the chunk text; empty for non-chunk vectors")
    value: float | None = Field(default=None, description="Numeric quantity extracted from the source (count, amount, year); None if not applicable")
    date: str | None = Field(default=None, description="ISO 8601 date extracted from the source; None if not applicable")

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v: Any) -> Any:
        return _coerce_none_string(v)

    @field_validator("date", mode="before")
    @classmethod
    def coerce_date(cls, v: Any) -> Any:
        return _coerce_none_string(v)


class AnalysisResult(BaseModel):
    data_points: List[DataPoint] = Field(default_factory=list)
