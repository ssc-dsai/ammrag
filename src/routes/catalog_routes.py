"""
Routes for catalogs and collections
"""

from typing import Optional

from fastapi import APIRouter, Query

from src.core.config import settings
from src.services.postgres_service import postgres_service

router = APIRouter(tags=["catalogs"])


@router.get("/catalogs", summary="List catalogs")
async def list_catalogs(
    id: Optional[int] = Query(None, description="Filter by catalog id"),
    name: Optional[str] = Query(None, description="Filter by catalog name"),
):
    catalogs = settings.catalogs
    if id is not None:
        catalogs = [c for c in catalogs if c.id == id]
    if name is not None:
        catalogs = [c for c in catalogs if c.name == name]
    return catalogs


@router.get("/collections", summary="List collections")
async def list_collections(
    id: Optional[int] = Query(None, description="Filter by collection id"),
    catalog_id: Optional[int] = Query(None, description="Filter by catalog id"),
):
    return postgres_service.get_collections(id=id, catalog_id=catalog_id)
