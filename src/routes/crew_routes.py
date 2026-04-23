"""
CrewAI routes for running the RAGFlow pipeline
"""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.schemas.crew_schemas import CrewQueryRequest, CrewQueryResponse
from src.services.crew_service import crew_service

router = APIRouter(
    prefix="/crews",
    tags=["crews"],
    responses={
        504: {"description": "RAGFlow timeout"},
        500: {"description": "RAGFlow error"},
    },
)


@router.post(
    "/query",
    response_model=CrewQueryResponse,
    summary="Ask a question",
    description="Run the RAGFlow pipeline to answer a natural language question: "
                "query decomposition → vector search → context widening → answer synthesis.",
)
async def crew_query(request: CrewQueryRequest) -> CrewQueryResponse:
    """
    Run the RAGFlow pipeline.

    Steps:
    1. ParsingCrew  — decompose question into focused sub-queries
    2. Qdrant search + neighbour expansion for each sub-query
    3. FormatCrew   — synthesise a Markdown answer with source references

    Args:
        request: CrewQueryRequest with the natural language question

    Returns:
        CrewQueryResponse with text answer, images, files, and sources
    """
    result = await crew_service.query(question=request.question, project_name=request.project_name, synthesis=request.synthesis)
    return CrewQueryResponse(**result)


@router.post(
    "/query/stream",
    summary="Ask a question (streaming)",
    description="Same as /query but streams Server-Sent Events: one progress event per pipeline stage, then a final result event.",
)
async def crew_query_stream(request: CrewQueryRequest) -> StreamingResponse:
    async def sse():
        async for event in crew_service.stream_query(request.question, request.project_name, request.synthesis):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
