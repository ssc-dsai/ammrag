"""
API routes package
"""

from fastapi import APIRouter

from .import_routes import router as import_router
from .retrieve_routes import router as retrieve_router
from .health_routes import router as health_router
from .ollama_routes import router as ollama_router
from .crew_routes import router as crew_router
from .search_routes import router as search_router
from .catalog_routes import router as catalog_router

# Aggregate all routers
api_router = APIRouter()

api_router.include_router(import_router)
api_router.include_router(retrieve_router)
api_router.include_router(health_router)
api_router.include_router(ollama_router)
api_router.include_router(crew_router)
api_router.include_router(search_router)
api_router.include_router(catalog_router)

__all__ = ["api_router"]
