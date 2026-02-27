"""
Import routes for file import operations
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import HttpUrl

from src.core.config import settings
from src.schemas.file_schemas import ImportResponse, BatchImportResponse
from src.services.index_service import index_service
from src.services.file_service import file_service

router = APIRouter(
    prefix="/import",
    tags=["import"],
    responses={
        400: {"description": "Bad request"},
        422: {"description": "Validation error"}
    }
)


@router.get(
    "/single",
    response_model=ImportResponse,
    summary="Import a single file",
    description="Import a single file from the provided URI and return a UUID"
)
async def import_single(
    uri: HttpUrl = Query(..., description="URI of the file to import")
) -> ImportResponse:
    """
    Import a single file from the provided URI
    
    Args:
        uri: The URI of the file to import
        
    Returns:
        ImportResponse with the generated UUID and import details
    """
    return await file_service.import_single_file(str(uri))


@router.get(
    "/batch",
    response_model=BatchImportResponse,
    summary="Import multiple files",
    description="Import multiple files from the provided URIs and return UUIDs for each"
)
async def import_batch(
    uri: List[HttpUrl] = Query(
        ...,
        description="List of file URIs to import (can be specified multiple times)"
    )
) -> BatchImportResponse:
    """
    Import multiple files from the provided URIs
    
    Args:
        uri: List of URIs of files to import (can be specified multiple times)
        
    Returns:
        BatchImportResponse with UUIDs for all imported files
    """
    uris = [str(u) for u in uri]
    return await file_service.import_batch_files(uris)


@router.get(
    "/catalog",
    response_model=BatchImportResponse,
    summary="Import files from catalog(s)",
    description="Import all files from a specific catalog or all catalogs",
)
async def import_catalog(
    catalog_id: Optional[int] = Query(None, description="Catalog id to import"),
    name: Optional[str] = Query(None, description="Catalog name to import"),
) -> BatchImportResponse:
    catalogs = index_service.catalogs
    if catalog_id is not None:
        catalogs = [c for c in catalogs if c.id == catalog_id]
    if name is not None:
        catalogs = [c for c in catalogs if c.name == name]

    if not catalogs:
        raise HTTPException(status_code=404, detail="Catalog not found")

    all_files: list[ImportResponse] = []
    for catalog in catalogs:
        result = await file_service.import_batch_files(catalog.path, catalog_id=catalog.id)
        all_files.extend(result.files)

    return BatchImportResponse(files=all_files, total=len(all_files))
