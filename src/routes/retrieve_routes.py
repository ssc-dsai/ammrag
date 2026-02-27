"""
Retrieve routes for file retrieval operations
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.schemas.file_schemas import RetrieveResponse
from src.services.file_service import file_service

router = APIRouter(
    tags=["retrieve"],
    responses={
        404: {"description": "File not found"}
    }
)


@router.get(
    "/retrieve/{file_uuid}",
    response_model=RetrieveResponse,
    summary="Retrieve file information",
    description="Retrieve file information and access link by UUID"
)
async def retrieve_file(file_uuid: str) -> RetrieveResponse:
    """
    Retrieve file information and access link by UUID
    
    Args:
        file_uuid: The UUID of the file to retrieve
        
    Returns:
        RetrieveResponse with file details and access link
        
    Raises:
        HTTPException: 404 if file not found
    """
    return file_service.retrieve_file(file_uuid)


@router.get(
    "/files/{file_uuid}/download",
    summary="Download file",
    description="Access endpoint for downloading the file"
)
async def download_file(file_uuid: str):
    """
    Download endpoint for accessing the file
    
    In production, this would:
    - Retrieve the file from storage
    - Return the actual file content
    - Or redirect to a CDN/storage service
    
    Args:
        file_uuid: The UUID of the file to download
        
    Returns:
        File content or redirect
        
    Raises:
        HTTPException: 404 if file not found
    """
    # Get file metadata to verify it exists
    file_metadata = file_service.get_file_metadata(file_uuid)
    
    # In production, you would:
    # 1. Fetch the actual file from storage
    # 2. Return it as a FileResponse or StreamingResponse
    # 3. Or redirect to a signed CDN URL
    
    return JSONResponse(
        content={
            "message": "File download endpoint",
            "uuid": file_uuid,
            "original_uri": file_metadata.uri,
            "note": "In production, this would return the actual file content or redirect to storage"
        }
    )
