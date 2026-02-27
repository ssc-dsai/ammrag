"""
Database models for file metadata

Note: Currently using in-memory storage. 
When migrating to a database, these models can be converted to SQLAlchemy/Tortoise ORM models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FileMetadata:
    """
    File metadata model
    
    In production with a database, this would be a SQLAlchemy model like:
    
    class FileMetadata(Base):
        __tablename__ = "files"
        
        uuid = Column(String, primary_key=True, index=True)
        uri = Column(String, nullable=False)
        imported_at = Column(DateTime, default=datetime.utcnow)
        file_size = Column(Integer, nullable=True)
        content_type = Column(String, nullable=True)
        status = Column(String, default="active")
    """
    
    uuid: str
    uri: str
    last_modified: str
    file_size: Optional[int] = None
    content_type: Optional[str] = None
    status: str = "active"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "uuid": self.uuid,
            "uri": self.uri,
            "last_modified": self.last_modified,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FileMetadata":
        """Create instance from dictionary"""
        return cls(**data)
