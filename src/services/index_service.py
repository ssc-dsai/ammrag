import logging
from dataclasses import dataclass, field
import os

import yaml

from src.models.catalog_model import CatalogItem
from src.models.vector_config_model import VectorConfig
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("config", {}) if isinstance(data, dict) else {}

@dataclass
class IndexService:
    # Load catalogs and doctypes from config file
    catalogs: list[CatalogItem] = field(default_factory=list)

    def __init__(self, config_path: str = os.getenv("CONFIG_PATH", "config/config.yml")):
        self.catalogs = []
        _config = load_config(config_path)

        for _entry in _config.get("catalogs", []):
            name = _entry.get("name")
            path = _entry.get("path")
            if not name or not path:
                continue
            vectors = [
                VectorConfig(
                    type=v.get("type", "dense"),
                    model=v.get("model"),
                    embedder=v.get("embedder"),
                    tokenizer=v.get("tokenizer"),
                    distance=v.get("distance"),
                )
                for v in _entry.get("vectors", [])
            ]
            metadata = _entry.get("metadata", {})
            if "location" in metadata:
                metadata["location"] = path.rstrip("/") + "/" + str(metadata["location"]).lstrip("/")
            self.catalogs.append(CatalogItem(name=name, path=path, vectors=vectors, metadata=metadata))
            logger.info("Loaded catalog '%s' from config (%d vectors)", name, len(vectors))

    def get_catalog(self, catalog_id: int) -> CatalogItem:
        """Return the catalog with the given id, or raise if not found."""
        catalog = next((c for c in self.catalogs if c.id == catalog_id), None)
        if catalog is None:
            raise ValueError(f"Catalog with id {catalog_id} not found")
        return catalog
    
index_service = IndexService()

if __name__ == "__main__":
    # For testing purposes, print loaded catalogs
    for catalog in index_service.catalogs:
        print(f"Catalog: {catalog.name}, Path: {catalog.path}, Vector configurations: {len(catalog.vector_configs)}")    


"""
Storage service for managing file metadata

This is the data access layer that handles storage operations.
"""

from typing import Dict, Optional, List
from datetime import datetime, timezone
import uuid as uuid_lib
import os
from urllib.parse import urlparse
from email.utils import parsedate_to_datetime
import httpx
from src.services.postgres_service import postgres_service
from src.models.file_model import FileMetadata

class FileIndex:
    """
    In-memory file storage service
    
    In production, this would be replaced with:
    - DatabaseStorage (using SQLAlchemy/Tortoise ORM)
    - S3Storage (for cloud storage)
    - RedisStorage (for caching)
    etc.
    """
    
    def __init__(self):
        self._index: Dict[str, FileMetadata] = {}
    
    async def create(self, uri: str) -> FileMetadata:
        """
        Create a new file record from URI info.

        Gathers metadata (last_modified, file_size, content_type) via a HEAD
        request for HTTP(S) URIs or from the local filesystem, then persists
        the record in the in-memory index.

        Args:
            uri: The URI of the file

        Returns:
            FileMetadata object with generated UUID
        """
        file_uuid = str(uuid_lib.uuid4())

        parsed = urlparse(uri)
        file_size = None
        content_type = None

        if parsed.scheme in ('http', 'https'):
            response = httpx.head(uri, follow_redirects=True)
            last_modified_header = response.headers.get('Last-Modified')
            if last_modified_header:
                modified_time = parsedate_to_datetime(last_modified_header)
            else:
                modified_time = datetime.now(timezone.utc)
            content_type = response.headers.get('Content-Type')
            content_length = response.headers.get('Content-Length')
            if content_length:
                file_size = int(content_length)
        else:
            file_path = parsed.path if parsed.scheme == 'file' else uri
            stat = os.stat(file_path)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            file_size = stat.st_size

        file_metadata = FileMetadata(
            uuid=file_uuid,
            uri=uri,
            last_modified=modified_time.isoformat(),
            file_size=file_size,
            content_type=content_type,
        )
        self._index[file_uuid] = file_metadata

        return file_metadata
    
    def get(self, file_uuid: str) -> Optional[FileMetadata]:
        """
        Retrieve file metadata by UUID
        
        Args:
            file_uuid: The UUID of the file
            
        Returns:
            FileMetadata if found, None otherwise
        """
        return self._index.get(file_uuid)
    
    def exists(self, file_uuid: str) -> bool:
        """
        Check if a file exists
        
        Args:
            file_uuid: The UUID of the file
            
        Returns:
            True if exists, False otherwise
        """
        return file_uuid in self._index
    
    def list_all(self) -> List[FileMetadata]:
        """
        List all files
        
        Returns:
            List of all FileMetadata objects
        """
        return list(self._index.values())
    
    def count(self) -> int:
        """
        Get total number of files
        
        Returns:
            Count of files in storage
        """
        return len(self._index)
    
    def delete(self, file_uuid: str) -> bool:
        """
        Delete a file record
        
        Args:
            file_uuid: The UUID of the file
            
        Returns:
            True if deleted, False if not found
        """
        if file_uuid in self._index:
            del self._index[file_uuid]
            return True
        return False
    
    def clear(self):
        """Clear all storage (useful for testing)"""
        self._index.clear()


# Global storage instance
# In production, this would be dependency-injected
file_index = FileIndex()


if __name__ == "__main__":
    # Simple test
    service = file_index
    file1 = service.create("http://192.168.68.92:8642/test/test.txt")
    
    print("All files:")
    for f in service.list_all():
        print(f.to_dict())
    
    print("Total files:", service.count())
    
    retrieved = service.get(file1.uuid)
    print("Retrieved file1:", retrieved.to_dict() if retrieved else "Not found")