"""
File service containing business logic for file operations
"""

import mimetypes
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin, unquote

import httpx
from fastapi import HTTPException

from email.utils import parsedate_to_datetime
from src.models.file_model import FileMetadata
from src.services.mmore_service import mmore_service
from src.schemas.file_schemas import (
    ImportResponse,
    BatchImportResponse,
    RetrieveResponse
)
from src.core.config import settings
from src.services.index_service import index_service
from src.services.postgres_service import postgres_service

import uuid as uuid_lib

class FileService:
    """
    Service layer for file operations
    
    Contains business logic separate from HTTP handling and data access
    """

    
    async def extract_file_metadata(self, uri: str) -> FileMetadata:
        """
        Gather metadata for a file from its URI.

        For HTTP(S) URIs a HEAD request is issued to obtain Last-Modified,
        Content-Type and Content-Length.  For local / file:// URIs the
        filesystem is queried via os.stat.

        Returns:
            A populated FileMetadata instance.
        """
        file_uuid = str(uuid_lib.uuid4())
        parsed = urlparse(uri)
        file_size = None
        content_type = None

        if parsed.scheme in ('http', 'https'):
            async with httpx.AsyncClient() as client:
                response = await client.head(uri, follow_redirects=True)
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
            modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            file_size = stat.st_size

        return FileMetadata(
            uuid=file_uuid,
            uri=uri,
            last_modified=modified_time.isoformat(),
            file_size=file_size,
            content_type=content_type,
        )

    async def import_single_file(
        self, uri: str, catalog_id: int | None = None
    ) -> ImportResponse:
        """
        Import a single file through the full ingest pipeline:
        1. Extract FileMetadata from URI info
        2. Download the file to a temporary location
        3. Process the temp file with mmore (text extraction)
        4. Record file in PostgreSQL and upsert vectors to Qdrant

        Args:
            uri: Local file path or HTTP/HTTPS URL
            catalog_id: Optional catalog id for catalog-based imports

        Returns:
            ImportResponse with file metadata
        """
        # 1. Extract metadata from URI
        file_metadata = await self.extract_file_metadata(uri)

        # 2. Download/copy file to a temporary location
        base_dir = settings.temp_file_path
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(dir=base_dir)
        try:
            tmp_path = await self._download_to_temp(uri, tmp_dir)

            # 3. Extract content via mmore (returns list of samples)
            samples = await mmore_service.process_file(tmp_path, tmp_dir=tmp_dir)

            # 4. Record file in PostgreSQL and upsert vectors to Qdrant
            file_type = (
                file_metadata.content_type
                or mimetypes.guess_type(uri)[0]
                or "unknown"
            )
            last_modified = datetime.fromisoformat(file_metadata.last_modified)

            is_image = file_type.startswith("image/")
            is_structured = any(
                file_type.endswith(t)
                for t in ("csv", "spreadsheetml.sheet", "vnd.ms-excel")
            ) or uri.lower().endswith((".csv", ".xlsx"))

            # Get location from URI
            parsed_uri = urlparse(uri)
            path_parts = [p for p in parsed_uri.path.strip("/").split("/") if p]
            location = unquote(path_parts[0]).replace("\xa0", " ") if path_parts else ""

            metadata = {
                "uri": uri,
                "uuid": file_metadata.uuid,
                "image": is_image,
                "structured": is_structured,
                "location": location,
            }

            catalog = next(
                (c for c in index_service.catalogs if c.id == catalog_id), None
            )
            if catalog is not None and catalog.collections:
                collection = catalog.collections[-1]
                for sample in samples:
                    if not sample.text:
                        continue

                    sample_meta = getattr(sample, "metadata", {}) or {}
                    sample_metadata = {**metadata}

                    # Build structured table name and include it in the payload
                    csv_data = sample_meta.get("csv_data")
                    table_number = sample_meta.get("table_number")
                    pg_table_name = None
                    if is_structured and csv_data and table_number is not None:
                        # Placeholder — file_id will be filled after add_file
                        sample_metadata["table_number"] = table_number

                    file_item = collection.add_file(
                        uri=uri,
                        file_type=file_type,
                        last_modified=last_modified,
                        text=sample.text,
                        metadata=sample_metadata,
                    )

                    # Store structured tables in Postgres
                    if is_structured and csv_data and file_item and file_item.id:
                        pg_table_name = f"file_{file_item.id}_table_{table_number or 1}"
                        await postgres_service.add_structured(
                            file_item.id, csv_data, table_name=pg_table_name
                        )

            return ImportResponse(
                uuid=file_metadata.uuid,
                uri=file_metadata.uri,
                last_modified=file_metadata.last_modified,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    async def import_batch_files(
        self, root_dir: str, catalog_id: int | None = None
    ) -> BatchImportResponse:
        """
        Import all files found under a root directory.

        Args:
            root_dir: A local filesystem path or a web URL pointing to a
                      directory.  Every file in every subdirectory will be
                      imported via import_single_file.
            catalog_id: Optional catalog id for catalog-based imports

        Returns:
            BatchImportResponse with list of imported files
        """
        parsed = urlparse(root_dir)

        if parsed.scheme in ('http', 'https'):
            uris = self._crawl_web_directory(root_dir)
        else:
            local_path = parsed.path if parsed.scheme == 'file' else root_dir
            uris = self._walk_local_directory(local_path)

        if not uris:
            raise HTTPException(
                status_code=400,
                detail="No files found in the provided directory"
            )

        imported_files = []
        for uri in uris:
            result = await self.import_single_file(uri, catalog_id=catalog_id)
            imported_files.append(result)

        return BatchImportResponse(
            files=imported_files,
            total=len(imported_files)
        )
    
    def retrieve_file(self, file_uuid: str) -> RetrieveResponse:
        """
        Retrieve file information by UUID
        
        Args:
            file_uuid: The UUID of the file
            
        Returns:
            RetrieveResponse with file metadata and access link
            
        Raises:
            HTTPException: If file not found
        """
        # Get file from storage
        file_metadata = self.index.get(file_uuid)
        
        if not file_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"File with UUID {file_uuid} not found"
            )
        
        # Generate access link
        access_link = self._generate_access_link(file_uuid)
        
        return RetrieveResponse(
            uuid=file_metadata.uuid,
            uri=file_metadata.uri,
            last_modified=file_metadata.last_modified,
            access_link=access_link
        )
    
    def get_file_metadata(self, file_uuid: str) -> FileMetadata:
        """
        Get raw file metadata
        
        Args:
            file_uuid: The UUID of the file
            
        Returns:
            FileMetadata object
            
        Raises:
            HTTPException: If file not found
        """
        file_metadata = self.index.get(file_uuid)
        
        if not file_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"File with UUID {file_uuid} not found"
            )
        
        return file_metadata
    
    async def _download_to_temp(self, uri: str, tmp_dir: str) -> str:
        """
        Download or copy a file into the given temporary directory.

        For HTTP(S) URIs the file is streamed down; for local/file URIs it is
        copied.  The original filename from the URI path is preserved so that
        mmore can auto-detect the file type from the extension.

        Returns:
            Absolute path to the file inside tmp_dir.
        """
        parsed = urlparse(uri)
        filename = os.path.basename(parsed.path) or "download"
        tmp_path = os.path.join(tmp_dir, filename)

        if parsed.scheme in ('http', 'https'):
            async with httpx.AsyncClient() as client:
                async with client.stream('GET', uri) as response:
                    response.raise_for_status()
                    with open(tmp_path, 'wb') as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
        else:
            local_path = parsed.path if parsed.scheme == 'file' else uri
            shutil.copy2(local_path, tmp_path)

        return tmp_path

    def _walk_local_directory(self, root_path: str) -> list[str]:
        """Recursively collect all file paths under a local directory."""
        uris: list[str] = []
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                uris.append(os.path.join(dirpath, filename))
        return uris

    def _crawl_web_directory(self, base_url: str) -> list[str]:
        """Crawl a web directory listing and collect all file URLs."""
        if not base_url.endswith('/'):
            base_url += '/'

        uris: list[str] = []
        visited: set[str] = set()
        self._crawl_recursive(base_url, base_url, uris, visited)
        return uris

    def _crawl_recursive(
        self, url: str, base_url: str, uris: list[str], visited: set[str]
    ) -> None:
        """Recursively follow directory links and collect file URLs."""
        if url in visited:
            return
        visited.add(url)

        try:
            response = httpx.get(url, timeout=30)
        except httpx.HTTPError:
            return
        if response.status_code != 200:
            return

        links = re.findall(r'href=["\']([^"\']+)["\']', response.text)

        for link in links:
            if link in ('../', '..', './', '.'):
                continue

            resolved = urljoin(url, link)

            if not resolved.startswith(base_url):
                continue

            if resolved.endswith('/'):
                self._crawl_recursive(resolved, base_url, uris, visited)
            elif resolved not in visited:
                visited.add(resolved)
                uris.append(resolved)

    def _generate_access_link(self, file_uuid: str) -> str:
        """
        Generate access link for file download
        
        In production, this could:
        - Generate signed URLs
        - Create CDN links
        - Generate temporary access tokens
        
        Args:
            file_uuid: The UUID of the file
            
        Returns:
            Access link string
        """
        return f"/files/{file_uuid}/download"
    
    def get_total_files(self) -> int:
        """
        Get total number of files in storage
        
        Returns:
            Count of files
        """
        return self.index.count()


# Global service instance
# In production, this would be dependency-injected
file_service = FileService()
