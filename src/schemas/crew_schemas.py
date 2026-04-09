"""
Pydantic schemas for CrewAI agent requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional

from src.agents.models.answer import QueryAspect


class CrewQueryRequest(BaseModel):
    """Request to run the RAGFlow pipeline."""
    question: str = Field(..., description="Natural language question to answer")
    project_name: Optional[str] = Field(None, description="Qdrant collection / project to search")


class CrewQueryResponse(BaseModel):
    """Structured response from the RAGFlow pipeline."""
    aspects: List[QueryAspect] = Field(
        default_factory=list,
        description="One entry per aspect of the answer, each with a discussion, quote, and source",
    )
    sources: List[str] = Field(default_factory=list, description="Deduplicated list of source URIs")
    images: List[str] = Field(default_factory=list, description="Image paths/URLs from results")
    files: List[str] = Field(default_factory=list, description="File paths/URLs from results")
