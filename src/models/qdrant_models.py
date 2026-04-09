from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, field_validator

# Payload models for Qdrant vector points

class QdrantPayload(BaseModel):
    """Base payload model for all Qdrant vector points."""

    type: str
    text: str
    uri: str

    def to_dict(self) -> dict:
        """Return all fields as a flat dict, skipping None values.
        Lists are preserved as-is; all other values are coerced to str."""
        result = {}
        for k, v in self.model_dump().items():
            if v is None:
                continue
            result[k] = v if isinstance(v, list) else str(v)
        return result


class ChunkPayload(QdrantPayload):
    """Payload for a text chunk vector point."""

    type: str = "chunk"
    chunk_id: int
    chunk_total: int


class FilePayload(QdrantPayload):
    """Payload for a file-level vector point."""

    type: str = "file"
    image: bool = False
    structured: bool = False
    last_modified: str = ""
    table_number: Optional[int] = None
    structured_tables: Optional[List[str]] = None

    @field_validator("last_modified", mode="before")
    @classmethod
    def coerce_last_modified(cls, v: Any) -> str:
        if isinstance(v, str):
            return v
        return v.isoformat() if hasattr(v, "isoformat") else str(v)


class DirectoryPayload(QdrantPayload):
    """Payload for a directory-level vector point."""

    type: str = "directory"


# Vector models for Qdrant results

class QdrantVector(BaseModel):
    collection_name: str
    point_id: str
    rank: int
    score: float
    payload: Any = {}


    def get_payload_field(self, key: str) -> Any:
        if isinstance(self.payload, dict):
            return self.payload.get(key)
        return getattr(self.payload, key, None)

    def get_type(self) -> str:
        return getattr(self.payload, "type")
    
    def get_uri(self) -> str | None:
        return getattr(self.payload, "uri", None) 
    
class ChunkVector(QdrantVector):
    """Vector point for a text chunk."""

    payload: ChunkPayload

    def get_file_vector(self) -> "FileVector":
        """Retrieve the parent FileVector for this chunk."""
        from src.services.qdrant_service import qdrant_service
        return qdrant_service.get_file_vector(self.collection_name, chunk=self)


class FileVector(QdrantVector):
    """Vector point for a file-level summary."""
    payload: FilePayload

    @property
    def table_name(self) -> Optional[str]:
        if self.payload.structured:
            return f"{self.collection_name}_file_{self.payload.uri}_table_{self.payload.table_number}"
        return None
    
    def get_directory_vector(self) -> "DirectoryVector":
        """Retrieve the parent DirectoryVector for this file."""
        from src.services.qdrant_service import qdrant_service
        return qdrant_service.get_directory_vector(self.collection_name, self)


class DirectoryVector(QdrantVector):
    """Vector point for a directory-level summary."""
    payload: DirectoryPayload
