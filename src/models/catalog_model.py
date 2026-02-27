from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid as uuid_lib

from src.models.file_model import FileMetadata
from src.models.vector_config_model import VectorConfig

logger = logging.getLogger(__name__)


@dataclass
class FileItem:
    """A file belonging to a collection, registered in PostgreSQL and Qdrant."""
    uri: str
    file_type: str
    last_modified: datetime
    metadata: FileMetadata
    id: Optional[int] = None


@dataclass
class CollectionItem:
    """A collection belonging to a catalog, registered in PostgreSQL and Qdrant."""
    catalog: CatalogItem
    name: Optional[str] = None
    id: Optional[int] = None    
    files: list[FileItem] = field(default_factory=list)

    def __init__(self, catalog: CatalogItem, name: Optional[str] = None, id: Optional[int] = None):
        """Create a new collection in both PostgreSQL and Qdrant."""
        
        from src.services.postgres_service import postgres_service
        from src.services.qdrant_service import qdrant_service

        self.catalog = catalog
        self.files = []
        self.name = name or f"{catalog.name}_{uuid_lib.uuid4().hex[:12]}"
        self.id = id or postgres_service.create_collection(catalog_id=catalog.id, name=self.name)

        # Ensure matching Qdrant collection exists
        if not qdrant_service.client.collection_exists(collection_name=self.name):
            qdrant_service.create_collection(self.name, self.catalog.vector_configs)

        logger.info("Collection '%s' (id=%s) ready for catalog '%s'",
                     self.name, self.id, self.catalog.name)
    


    def add_file(
        self,
        uri: str,
        file_type: str,
        last_modified: datetime,
        text: str,
        metadata: Dict[str, Any],
        processed: bool = False,
    ) -> Optional[FileItem]:
        """Record a file in PostgreSQL, upsert its content into Qdrant, and return a FileItem.

        Deduplication rules:
        - Same URI and same last_modified → skip entirely, return None.
        - Same URI but different last_modified → update timestamp in Postgres,
          delete old vectors in Qdrant, re-chunk-and-upsert, return FileItem.
        - New URI → insert into Postgres and Qdrant, return FileItem.
        """
        from src.services.postgres_service import postgres_service
        from src.services.qdrant_service import qdrant_service

        existing = postgres_service.get_file(catalog_id=self.id, uri=uri)

        if existing is not None:
            if str(existing["last_modified"]) == str(last_modified):
                logger.info("Skipping unchanged file '%s'", uri)
                return None

            # Timestamp changed → update Postgres and re-index in Qdrant
            file_id = existing["id"]
            postgres_service.update_file_timestamp(file_id, last_modified)
            qdrant_service.delete_by_file_id(self.name, file_id)
            logger.info("Updating changed file '%s' (id=%d)", uri, file_id)
        else:
            file_id = postgres_service.add_file(
                catalog_id=self.id,
                uri=uri,
                file_type=file_type,
                last_modified=last_modified,
                processed=processed,
            )

        if file_id is not None and text:
            payload = {**metadata, "file_id": file_id}
            if "table_number" in metadata:
                payload["pg_table_name"] = f"file_{file_id}_table_{metadata['table_number']}"
            qdrant_service.chunk_and_upsert(
                text=text,
                metadata=payload,
                collection_name=self.name,
            )

        file_item = FileItem(
            uri=uri,
            file_type=file_type,
            last_modified=last_modified,
            metadata=FileMetadata(
                uuid=metadata.get("uuid", ""),
                uri=uri,
                last_modified=last_modified.isoformat(),
                content_type=file_type,
            ),
            id=file_id,
        )
        self.files.append(file_item)

        return file_item


@dataclass
class CatalogItem:
    """A catalog loaded from config and registered in PostgreSQL."""
    name: str
    path: str
    id: Optional[int] = None
    collections: list[CollectionItem] = field(default_factory=list)
    vector_configs: list[VectorConfig] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, path: str, vectors: Optional[list[VectorConfig]] = None, metadata: Optional[Dict[str, Any]] = None):

        from src.services.postgres_service import postgres_service
        from src.services.qdrant_service import qdrant_service

        self.name = name
        self.path = path
        self.vector_configs = vectors or []
        self.metadata = metadata or {}
        self.collections = []

        self.id = postgres_service.add_catalog(self.name, self.path)
        rows = postgres_service.get_collections(catalog_id=self.id)

        if not rows:
            self.add_collection()
        else:
            self.collections = [
                CollectionItem(catalog=self, name=row["name"], id=row["id"])
                for row in rows
            ]

        logger.info(
            "Registered catalog '%s' (id=%d, %d collections)",
            self.name, self.id, len(self.collections),
        )

    @property
    def collection(self) -> Optional[CollectionItem]:
        """Return the first collection for this catalog, if any."""
        return self.collections[0] if self.collections else None


    def add_collection(self) -> CollectionItem:
        """Create and register a new collection for this catalog."""
        collection = CollectionItem(catalog=self)
        self.collections.append(collection)
        return collection



@dataclass
class DocTypeItem:
    """A document type loaded from config."""
    name: str


# if __name__ == "__main__":
#     # Example usage
#     catalog = CatalogItem(name="Example Catalog", path="/data/example", vectors=[
#         VectorConfig(type="dense", model="sentence-transformers/all-MiniLM-L6-v2")
#     ])
#     collection = catalog.collection
#     if collection:
#         collection.add_file(
#             uri="http://example.com/file1.txt",
#             file_type="text/plain",
#             last_modified=datetime.utcnow(),
#             text="This is an example file content.",
#             metadata={"source": "example"}
#         )
#     else:        logger.error("No collection available for catalog '%s'", catalog.name)

