"""
Pydantic schemas for request and response validation
"""

from pydantic import BaseModel, Field, HttpUrl
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


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    services: dict = Field(default_factory=dict, description="Per-service availability")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "ollama": True,
                    "qdrant": True,
                    "postgres": True,
                }
            }
        }


class ImportJobResponse(BaseModel):
    """Status of a background import job"""
    status: str = Field(..., description="idle | running | completed | failed")
    projects: List[str] = Field(default_factory=list, description="Project(s) being / last imported")
    started_at: datetime | None = Field(None, description="When the import started")
    completed_at: datetime | None = Field(None, description="When the import finished")
    files_total: int = Field(0, description="Total files to process")
    files_processed: int = Field(0, description="Files processed so far")
    files: List[ImportResponse] = Field(default_factory=list, description="Results (when completed)")
    error: str | None = Field(None, description="Error message if failed")


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "File not found"
            }
        }
