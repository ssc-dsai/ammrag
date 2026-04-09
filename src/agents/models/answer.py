from pydantic import BaseModel, Field
from typing import List


class QueryAspect(BaseModel):
    """One aspect of the answer to the original query."""
    discussion: str = Field(description="Brief explanation of this aspect of the answer")
    quote: str = Field(description="Verbatim or near-verbatim quote from the source document that supports this aspect")
    source: str = Field(description="URI or path of the source document the quote was taken from")


class QueryAnswer(BaseModel):
    """Structured answer composed of one item per aspect of the original query."""
    aspects: List[QueryAspect] = Field(description="One entry per aspect required to fully answer the query")
