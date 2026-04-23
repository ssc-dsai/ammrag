"""
Pydantic schemas for CrewAI agent requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class CrewQueryRequest(BaseModel):
    """Request to run the RAGFlow pipeline."""
    question: str = Field(..., description="Natural language question to answer")
    project_name: Optional[str] = Field(None, description="Qdrant collection / project to search")
    synthesis: bool = Field(default=False, description="If true, run FormatCrew to produce a markdown answer")


class CrewQueryResponse(BaseModel):
    """Structured response from the RAGFlow pipeline."""
    answer: str = Field(default="", description="Pre-formatted markdown answer with inline citations")
    sources: List[str] = Field(default_factory=list, description="Deduplicated list of source URIs")
    images: List[str] = Field(default_factory=list, description="Image paths/URLs from results")
    files: List[str] = Field(default_factory=list, description="File paths/URLs from results")
