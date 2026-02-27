"""
Search routes for vector and metadata search operations
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict

from src.schemas.search_schemas import (
    VectorSearchRequest,
    VectorSearchResponse,
    MetadataSearchRequest,
    MetadataSearchResult,
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    SQLQueryRequest,
)
from src.services.qdrant_service import qdrant_service
from src.services.postgres_service import postgres_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={
        404: {"description": "Document not found"},
        500: {"description": "Search service error"},
    },
)


@router.post(
    "/vectors",
    response_model=VectorSearchResponse,
    summary="Semantic vector search",
    description="Search documents using semantic similarity via Qdrant",
)
async def vector_search(request: VectorSearchRequest) -> Dict:
    """
    Perform semantic vector search.

    Embeds the query text and searches the Qdrant collection for
    similar document chunks, returning results ranked by relevance.

    Args:
        request: VectorSearchRequest with query and search parameters

    Returns:
        VectorSearchResponse with ranked results
    """
    try:
        return await qdrant_service.vector_search(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            score_threshold=request.score_threshold,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("vector_search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vector search error: {e}")


@router.post(
    "/metadata",
    response_model=MetadataSearchResult,
    summary="Metadata search",
    description="Query PostgreSQL metadata tables",
)
async def metadata_search(request: MetadataSearchRequest) -> Dict:
    """
    Search PostgreSQL metadata tables.

    Queries structured metadata with optional text search and
    key-value filters.

    Args:
        request: MetadataSearchRequest with query parameters

    Returns:
        MetadataSearchResult with columns and rows
    """
    try:
        return await postgres_service.search_metadata(
            query=request.query,
            table_name=request.table_name,
            filters=request.filters,
            limit=request.limit,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("metadata_search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Metadata search error: {e}")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query nodes from Qdrant",
    description="Retrieve nodes from Qdrant based on a semantic query with optional payload filters",
)
async def query_nodes(request: QueryRequest) -> QueryResponse:
    """
    Query Qdrant for similar document nodes.

    Embeds the query text, performs nearest-neighbour search on the dense
    vector, and returns matching nodes with all payload metadata.
    """
    results = qdrant_service.query(
        query_text=request.query,
        collection_name=request.collection_name,
        limit=request.limit,
        score_threshold=request.score_threshold,
        filter_conditions=request.filter_conditions,
    )

    return QueryResponse(
        results=[
            QueryResult(
                id=str(r["id"]),
                score=r["score"],
                payload={k: v for k, v in r.items() if k not in ("id", "score")},
            )
            for r in results
        ],
        total=len(results),
        query=request.query,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Get document by ID",
    description="Retrieve a specific document from the vector store by its ID",
)
async def get_document(document_id: str) -> Dict:
    """
    Retrieve a document by its vector store ID.

    Args:
        document_id: The Qdrant point ID

    Returns:
        DocumentResponse with content and metadata
    """
    return await qdrant_service.get_document(point_id=document_id)


@router.post(
    "/sql",
    response_model=MetadataSearchResult,
    summary="Execute SQL query",
    description="Execute a read-only SELECT query against PostgreSQL structured data tables",
)
async def execute_sql(request: SQLQueryRequest) -> Dict:
    """
    Execute a read-only SQL SELECT query.

    Only SELECT statements are allowed. Used by the retrieval crew to
    query structured data tables discovered during vector search.

    Args:
        request: SQLQueryRequest with the SQL query

    Returns:
        MetadataSearchResult with columns and rows
    """
    try:
        return await postgres_service.execute_select(sql_query=request.sql_query)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("execute_sql failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"SQL execution error: {e}")
