"""
Pydantic schemas for CrewAI agent requests and responses
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional


class CrewQueryRequest(BaseModel):
    """Request to run the query retrieval crew.

    At least one of 'name' or 'catalog_id' must be provided to identify
    which catalog's collections to search.
    """
    question: str = Field(..., description="Natural language question to answer")
    name: Optional[str] = Field(None, description="Catalog name to query")
    catalog_id: Optional[int] = Field(None, description="Catalog ID to query")

    @model_validator(mode="after")
    def require_catalog_identifier(self):
        if not self.name and self.catalog_id is None:
            raise ValueError("At least one of 'name' or 'catalog_id' must be provided")
        return self


class CrewQueryResponse(BaseModel):
    """Structured response from the query retrieval crew"""
    text: str = Field(..., description="Natural language answer (Markdown)")
    images: List[str] = Field(default_factory=list, description="Image paths/URLs from results")
    files: List[str] = Field(default_factory=list, description="File paths/URLs from results")
    sources: List[str] = Field(default_factory=list, description="Source references")
