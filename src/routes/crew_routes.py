"""
CrewAI routes for running the query retrieval crew
"""

from fastapi import APIRouter, HTTPException

from src.schemas.crew_schemas import CrewQueryRequest, CrewQueryResponse
from src.services.crew_service import crew_service
from src.services.postgres_service import postgres_service

router = APIRouter(
    prefix="/crews",
    tags=["crews"],
    responses={
        504: {"description": "Crew execution timeout"},
        500: {"description": "Crew execution error"},
    },
)


@router.post(
    "/query",
    response_model=CrewQueryResponse,
    summary="Ask a question",
    description="Run the QueryRetrievalCrew to answer a natural language question "
                "using vector search and metadata lookup over the indexed corpus. "
                "Requires at least one of 'name' or 'catalog_id' to identify the catalog.",
)
async def crew_query(request: CrewQueryRequest) -> CrewQueryResponse:
    """
    Run the query retrieval crew.

    Resolves the catalog's collections, then runs the crew pipeline:
    1. RAG Retriever — vector search across all collections
    2. Structured Data Analyst — detect and query structured tables
    3. Response Compiler — compile Markdown response with source URIs

    Args:
        request: CrewQueryRequest with question and catalog identifier

    Returns:
        CrewQueryResponse with text answer, images, files, and sources
    """
    # Resolve catalog
    catalogs = postgres_service.get_catalogs(
        id=request.catalog_id, name=request.name
    )
    if not catalogs:
        raise HTTPException(
            status_code=404,
            detail="No catalog found matching the provided name or catalog_id",
        )

    catalog = catalogs[0]
    collections = postgres_service.get_collections(catalog_id=catalog["id"])
    if not collections:
        raise HTTPException(
            status_code=404,
            detail="No collections found for the specified catalog",
        )

    collection_names = [c["name"] for c in collections]

    result = await crew_service.query(
        question=request.question,
        collection_names=collection_names,
    )
    return CrewQueryResponse(**result)
