"""
Health check and utility routes
"""

from fastapi import APIRouter

from src.schemas.file_schemas import HealthResponse
from src.services.file_service import file_service
from src.core.config import settings

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns:
        HealthResponse with service status and metrics
    """
    return HealthResponse(
        status="healthy",
        total_files=file_service.get_total_files()
    )


@router.get(
    "/",
    summary="Root endpoint",
    description="API information and available endpoints"
)
async def root():
    """
    Root endpoint with API information
    
    Returns:
        Dictionary with API information and available endpoints
    """
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "endpoints": {
            "import_single": "/import/single?uri=<file_uri>",
            "import_batch": "/import/batch?uri=<file_uri1>&uri=<file_uri2>...",
            "retrieve": "/retrieve/{file_uuid}",
            "download": "/files/{file_uuid}/download",
            "health": "/health",
            "docs": "/docs"
        }
    }
