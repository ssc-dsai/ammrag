"""
API routes package
"""

from fastapi import APIRouter

from .import_routes import router as import_router
from .health_routes import router as health_router
from .crew_routes import router as crew_router

# Aggregate all routers
api_router = APIRouter()

api_router.include_router(import_router)
api_router.include_router(health_router)
api_router.include_router(crew_router)

__all__ = ["api_router"]
