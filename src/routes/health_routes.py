"""
Health check and utility routes
"""

import logging

import httpx

from fastapi import APIRouter

from src.schemas.file_schemas import HealthResponse
from src.services.file_service import file_service
from src.core.config import settings

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


async def _check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.ollama_host}/api/tags")
            return r.status_code == 200
    except httpx.TimeoutException:
        logger.warning("Ollama health check timed out (%s)", settings.ollama_host)
        return False
    except httpx.ConnectError:
        logger.warning("Ollama unreachable (%s)", settings.ollama_host)
        return False
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        return False


def _check_qdrant() -> bool:
    try:
        from src.services.qdrant_service import qdrant_service
        qdrant_service.client.get_collections()
        return True
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return False


def _check_postgres() -> bool:
    try:
        from src.services.postgres_service import postgres_service
        conn = postgres_service._get_connection()
        conn.close()
        return True
    except Exception as exc:
        logger.warning("Postgres health check failed: %s", exc)
        return False


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service and its dependencies"
)
async def health_check() -> HealthResponse:
    import asyncio

    ollama_ok, qdrant_ok, postgres_ok = await asyncio.gather(
        _check_ollama(),
        asyncio.to_thread(_check_qdrant),
        asyncio.to_thread(_check_postgres),
    )

    all_ok = ollama_ok and qdrant_ok and postgres_ok
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        services={
            "ollama": ollama_ok,
            "qdrant": qdrant_ok,
            "postgres": postgres_ok,
        },
    )


@router.get(
    "/",
    summary="Root endpoint",
    description="API information and available endpoints"
)
async def root():
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
