from typing import List, Literal

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    uri: str = Field(description="URI of the source file")
    type: Literal["image", "text"] = Field(description="Whether this data point is an image or text")
    pertinent: bool = Field(description="Whether this vector is pertinent to the step")
    step: str = Field(description="The plan step (info field) this was assessed against")
    reasoning: str = Field(description="Why the vector is or is not pertinent to the step")
    quote: str = Field(default="", description="Verbatim excerpt from the chunk text; empty for non-chunk vectors")


class AnalysisResult(BaseModel):
    data_points: List[DataPoint] = Field(default_factory=list)
