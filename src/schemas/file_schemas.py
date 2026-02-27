"""
Pydantic schemas for request and response validation
"""

from pydantic import BaseModel, HttpUrl
from typing import List
from datetime import datetime


class ImportResponse(BaseModel):
    """Response model for file import operations"""
    uuid: str
    uri: str
    last_modified: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "uri": "https://example.com/document.pdf",
                "imported_at": "2024-02-02T10:30:00.123456"
            }
        }


class BatchImportResponse(BaseModel):
    """Response model for batch import operations"""
    files: List[ImportResponse]
    total: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "files": [
                    {
                        "uuid": "123e4567-e89b-12d3-a456-426614174000",
                        "uri": "https://example.com/doc1.pdf",
                        "imported_at": "2024-02-02T10:30:00.123456"
                    }
                ],
                "total": 1
            }
        }


class RetrieveResponse(BaseModel):
    """Response model for file retrieval"""
    uuid: str
    uri: str
    last_modified: str
    access_link: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "uri": "https://example.com/document.pdf",
                "imported_at": "2024-02-02T10:30:00.123456",
                "access_link": "/files/123e4567-e89b-12d3-a456-426614174000/download"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    total_files: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "total_files": 42
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "File not found"
            }
        }
