"""
Pydantic schemas for search requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class VectorSearchRequest(BaseModel):
    """Request for semantic vector search"""
    query: str = Field(..., description="Natural language search query")
    collection_name: Optional[str] = Field(
        None, description="Qdrant collection to search (defaults to configured collection)"
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    score_threshold: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class VectorSearchResult(BaseModel):
    """Single vector search result"""
    id: str = Field(..., description="Point ID in Qdrant")
    score: float = Field(..., description="Similarity score")
    content: Optional[str] = Field(None, description="Text content of the chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Associated metadata")


class VectorSearchResponse(BaseModel):
    """Response from vector search"""
    results: List[VectorSearchResult] = Field(
        ..., description="Search results ordered by relevance"
    )
    total: int = Field(..., description="Number of results returned")
    query: str = Field(..., description="Original query")


class MetadataSearchRequest(BaseModel):
    """Request for PostgreSQL metadata search"""
    query: Optional[str] = Field(
        None, description="Text search term for metadata"
    )
    table_name: Optional[str] = Field(
        None, description="Specific table to query"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Key-value filters for metadata"
    )
    limit: int = Field(50, ge=1, le=500, description="Maximum results to return")


class MetadataSearchResult(BaseModel):
    """Metadata search result"""
    columns: List[str] = Field(..., description="Column names")
    rows: List[List[Any]] = Field(..., description="Row data")
    row_count: int = Field(..., description="Number of rows returned")


class DocumentResponse(BaseModel):
    """Response for individual document retrieval"""
    id: str = Field(..., description="Document ID")
    content: Optional[str] = Field(None, description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    source: Optional[str] = Field(None, description="Source/origin of the document")


class QueryRequest(BaseModel):
    """Request for Qdrant node retrieval query"""
    query: str = Field(..., description="Natural language search query")
    collection_name: str = Field(..., description="Qdrant collection to search")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    score_threshold: Optional[float] = Field(
        None, description="Minimum similarity score (None = no threshold)"
    )
    filter_conditions: Optional[Dict[str, Any]] = Field(
        None, description="Payload filters as key-value pairs"
    )


class QueryResult(BaseModel):
    """Single query result with all payload fields"""
    id: str = Field(..., description="Point ID in Qdrant")
    score: float = Field(..., description="Similarity score")
    payload: Dict[str, Any] = Field(default_factory=dict, description="All payload fields")


class QueryResponse(BaseModel):
    """Response from node retrieval query"""
    results: List[QueryResult] = Field(..., description="Search results ordered by relevance")
    total: int = Field(..., description="Number of results returned")
    query: str = Field(..., description="Original query")


class SQLQueryRequest(BaseModel):
    """Request for executing a read-only SQL query"""
    sql_query: str = Field(..., description="SQL SELECT query to execute")


class QueryDocumentsResponse(BaseModel):
    """Structured response from the query_documents MCP tool / retrieval crew"""
    text: str = Field(..., description="Natural language response to the query")
    images: List[str] = Field(
        default_factory=list, description="List of image URLs/paths"
    )
    files: List[str] = Field(
        default_factory=list, description="List of file URLs/paths"
    )
    sources: List[str] = Field(
        default_factory=list, description="Source references used"
    )
