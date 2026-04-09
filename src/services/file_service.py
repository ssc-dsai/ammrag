"""
File service containing business logic for file operations
"""

import asyncio
import mimetypes
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urljoin, unquote

import httpx
from fastapi import HTTPException

from email.utils import parsedate_to_datetime
from src.models.file_model import FileMetadata
from src.models.qdrant_models import FilePayload
from src.services.mmore_service import mmore_service
from src.schemas.file_schemas import (
    ImportResponse,
    BatchImportResponse
)
from src.core.config import settings
from src.services.postgres_service import postgres_service
from src.services.qdrant_service import qdrant_service
from src.services.ollama_service import ollama_service, get_prompt_config
from src.agents.flows.summarize import SummarizeFlow

import logging
logger = logging.getLogger(__name__)

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

    async def delete_file(self, collection_name: str, uri: str):
        existing = qdrant_service.get_point_by_uri(collection_name=collection_name, uri=uri)
        if existing is not None and existing.get_payload_field("structured"):
            postgres_service.delete_structured_by_file_id(uri)
        qdrant_service.delete_point_by_uri(collection_name=collection_name, uri=uri)
        qdrant_service.delete_directory_by_uri(collection_name=collection_name, uri=uri)

    async def import_single_file(
        self, uri: str, collection: str, _update_dirs: bool = True
    ) -> ImportResponse:
        """
        Import a single file through the full ingest pipeline:
        1. Extract FileMetadata from URI info
        2. Download the file to a temporary location
        3. Process the temp file with mmore (text extraction)
        4. Record file in PostgreSQL and upsert vectors to Qdrant

        Args:
            uri: Local file path or HTTP/HTTPS URL
            project_name: Name of the project to import the file

        Returns:
            ImportResponse with file metadata
        """
        # Extract metadata from URI
        file_metadata = await self.extract_file_metadata(uri)

        # Determine if the file has changed since last import by comparing last_modified to existing Qdrant point
        existing = qdrant_service.get_point_by_uri(collection_name=collection, uri=uri)

        if existing is not None:
            if str(existing.get_payload_field("last_modified")) == str(file_metadata.last_modified):
                logger.info("Skipping unchanged file '%s'", uri)
                return ImportResponse(uuid=existing.point_id, uri=uri, last_modified=file_metadata.last_modified)

            # Timestamp changed → delete and re-index
            await self.delete_file(collection, uri)

            logger.info("Updating changed file '%s'", uri)
        # 2. Download/copy file to a temporary location
        base_dir = settings.temp_file_path
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=base_dir) as tmp_dir:

                tmp_path = await self._download_to_temp(uri, tmp_dir)

                # Determine if file is an image or structured data based on content type and/or extension
                file_type = (
                    file_metadata.content_type
                    or mimetypes.guess_type(uri)[0]
                    or "unknown"
                )

                is_image = file_type.startswith("image/")
                is_xlsx = uri.lower().endswith((".xlsx", ".xls")) or any(
                    file_type.endswith(t) for t in ("spreadsheetml.sheet", "vnd.ms-excel")
                )
                is_structured = is_xlsx or any(
                    file_type.endswith(t) for t in ("csv",)
                ) or uri.lower().endswith(".csv")



                samples = await mmore_service.process_file(tmp_path, tmp_dir=tmp_dir)
                for sample in samples:
                    if not sample.text:
                        continue
                    sample_meta = getattr(sample, "metadata", {}) or {}
                    csv_data = sample_meta.get("csv_data") or sample.text
                
                    all_csv_parts: list[str] = []
                    all_table_ids: list[str] = []
                    if is_structured and csv_data:
                        sheet_csvs = mmore_service.export_xlsx_sheet_csvs(tmp_path, tmp_dir)
                        for _, csv_path in sheet_csvs:
                            with open(csv_path) as f:
                                csv_data = f.read()
                            all_csv_parts.append(csv_data)
                            all_table_ids.extend(await postgres_service.add_structured(file_id=uri, csv_data=csv_data))

                    file_payload = FilePayload(
                        text=sample.text,
                        uri=uri,
                        image=is_image,
                        structured=is_structured,
                        last_modified=file_metadata.last_modified,
                        structured_tables=all_table_ids or None,
                    )
                    await self.add_file(collection_name=collection, file_payload=file_payload)


                # if is_xlsx:
                #     sheet_csvs = mmore_service.export_xlsx_sheet_csvs(tmp_path, tmp_dir)
                #     all_csv_parts: list[str] = []
                #     all_table_ids: list[str] = []
                #     for _, csv_path in sheet_csvs:
                #         with open(csv_path) as f:
                #             csv_data = f.read()
                #         all_csv_parts.append(csv_data)
                #         all_table_ids.extend(await postgres_service.add_structured(file_id=uri, csv_data=csv_data))
                #     await self.add_file(collection_name=collection, file_payload=FilePayload(
                #         text="\n".join(all_csv_parts),
                #         uri=uri,
                #         image=False,
                #         structured=True,
                #         last_modified=file_metadata.last_modified,
                #         structured_tables=all_table_ids or None,
                #     ))
                # else:
                #     samples = await mmore_service.process_file(tmp_path, tmp_dir=tmp_dir)
                #     for sample in samples:
                #         if not sample.text:
                #             continue
                #         sample_meta = getattr(sample, "metadata", {}) or {}
                #         csv_data = sample_meta.get("csv_data") or sample.text
                #         table_ids: list[str] = []
                #         if is_structured and csv_data:
                #             table_ids = await postgres_service.add_structured(
                #                 file_id=uri, csv_data=csv_data
                #             )
                #         file_payload = FilePayload(
                #             text=sample.text,
                #             uri=uri,
                #             image=is_image,
                #             structured=is_structured,
                #             last_modified=file_metadata.last_modified,
                #             structured_tables=table_ids or None,
                #         )
                #         await self.add_file(collection_name=collection, file_payload=file_payload)

            if _update_dirs:
                await self._update_directory_tree(collection, [uri])

            return ImportResponse(
                uuid=file_metadata.uuid,
                uri=file_metadata.uri,
                last_modified=file_metadata.last_modified,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    

    async def add_file(
        self,
        collection_name: str,
        file_payload: FilePayload
    ) -> str | None:
        """Upsert a file into Qdrant. Returns the file UUID, or None if skipped."""
        text = file_payload.to_dict().get("text") or ""

        if file_payload.image:
            # Image description was already generated by mmore_service._describe_image
            # and stored in file_payload.text — use it directly.
            description = text
        else:
            # 1. Try Ollama directly (fast path)
            description = ""
            _NETWORK_STATUSES = {503, 504}

            try:
                result = await ollama_service.generate(
                    **get_prompt_config("summarize_file", text=text[:3000])
                )
                description = result.response.strip()
            except HTTPException as exc:
                if exc.status_code in _NETWORK_STATUSES:
                    raise
                logger.warning("Ollama HTTP error (status=%d) for '%s', falling back to SummarizeFlow", exc.status_code, file_payload.uri)
            except Exception as exc:
                logger.warning("Ollama generate failed for '%s': %s — falling back to SummarizeFlow", file_payload.uri, exc)

            # 2. Fall back to SummarizeFlow if Ollama returned empty (e.g. done_reason: "length")
            if not description:
                try:
                    flow = SummarizeFlow()
                    await asyncio.to_thread(flow.kickoff, inputs={"text": text})
                    description = flow.state.get("condensed_summary", "").strip()
                except Exception as exc:
                    logger.warning("SummarizeFlow failed for '%s': %s — using text truncation", file_payload.uri, exc)

            if not description:
                description = text[:384]

        if text:
            if not file_payload.image:
                qdrant_service.upsert_chunk_points(
                    collection_name=collection_name,
                    payload=file_payload
                )

            qdrant_service.upsert_file_point(
                collection_name=collection_name,
                description=description,
                payload=file_payload
            )


        return file_payload.uri


    async def import_batch_files(
        self,
        root_dir: str,
        project_name: str | None = None,
        catalog_id: int | None = None,
        on_file_imported=None,
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

        collection = project_name or settings.qdrant_collection_name
        qdrant_service.ensure_collection(collection)

        # Detect removed files by comparing current crawl against the stored filelist
        current_uri_set = set(uris)
        previous_uris = postgres_service.get_filelist(root_dir, collection)
        if not previous_uris:
            # Seed the filelist from Qdrant for this root_dir prefix
            previous_uris = [
                u for u in qdrant_service.get_all_uris(collection, point_type="file")
                if u.startswith(root_dir.rstrip("/"))
            ]
            logger.info("Seeded filelist for '%s' with %d existing URI(s)", root_dir, len(previous_uris))

        removed_uris = [u for u in previous_uris if u not in current_uri_set]
        if removed_uris:
            logger.info("Removing %d file(s) no longer present under '%s'", len(removed_uris), root_dir)
            for uri in removed_uris:
                try:
                    await self.delete_file(collection, uri)
                except Exception as exc:
                    logger.warning("Failed to delete removed file '%s': %s", uri, exc)

        imported_files = []
        imported_uris = []
        failed_uris = []
        for uri in uris:
            try:
                result = await self.import_single_file(uri, collection=collection, _update_dirs=False)
                imported_files.append(result)
                imported_uris.append(uri)
                if on_file_imported is not None:
                    on_file_imported(result)
            except Exception as exc:
                logger.error("Failed to import '%s': %s — skipping", uri, exc)
                failed_uris.append(uri)

        if failed_uris:
            logger.warning("%d file(s) failed to import: %s", len(failed_uris), failed_uris)

        await self._update_directory_tree(collection, imported_uris)

        # Persist the current filelist (successful imports only) for next run
        postgres_service.set_filelist(root_dir, collection, list(current_uri_set))

        return BatchImportResponse(
            files=imported_files,
            total=len(imported_files)
        )
    
    def retrieve_file(self, identifier: str, collection_name: str | None = None):
        """Retrieve a file-level QdrantVector by point ID or URI.

        Args:
            identifier: Qdrant point UUID or file URI
            collection_name: Qdrant collection to search (defaults to settings)

        Returns:
            QdrantVector for the matching point

        Raises:
            HTTPException: If not found
        """
        collection = collection_name or settings.qdrant_collection_name
        return qdrant_service.get_point(collection, identifier)
    
    @staticmethod
    def _parent_dir_uri(uri: str) -> str | None:
        """Return the parent directory URI (no trailing slash), or None if at root."""
        parsed = urlparse(uri)
        if parsed.scheme in ("http", "https"):
            path = parsed.path.rstrip("/")
            if not path or path == "/":
                return None
            parent_path = path.rsplit("/", 1)[0]
            if not parent_path:
                return None
            return f"{parsed.scheme}://{parsed.netloc}{parent_path}"
        else:
            parent = os.path.dirname(uri.rstrip("/"))
            return parent if parent and parent != uri.rstrip("/") else None

    async def _update_directory_tree(
        self, collection_name: str, file_uris: list[str]
    ) -> None:
        """Update directory-type points bottom-up for all ancestors of the given file URIs."""
        from collections import defaultdict

        file_uris_set = set(file_uris)
        dir_children: dict[str, set[str]] = defaultdict(set)

        # Build directory hierarchy from file URIs
        for file_uri in file_uris:
            current = file_uri
            while True:
                parent = self._parent_dir_uri(current)
                if parent is None:
                    break
                dir_children[parent].add(current)
                current = parent

        if not dir_children:
            return

        # Process deepest directories first
        dirs_sorted = sorted(dir_children, key=lambda u: u.count("/"), reverse=True)

        dir_descriptions: dict[str, str] = {}
        for dir_uri in dirs_sorted:
            children = dir_children[dir_uri]
            file_children = [c for c in children if c in file_uris_set]
            dir_child_uris = [c for c in children if c not in file_uris_set]

            # Fetch file descriptions from Qdrant
            descs = qdrant_service.get_point_descriptions(collection_name=collection_name, uris=file_children, point_type="file")
            # Use locally computed subdirectory descriptions (already processed, since bottom-up)
            descs.update({u: dir_descriptions[u] for u in dir_child_uris if u in dir_descriptions})

            if not descs:
                continue

            combined = "\n".join(f"- {d}" for d in list(descs.values())[:20])
            try:
                result = await ollama_service.generate(
                    **get_prompt_config("summarize_directory", uri=unquote(dir_uri), content=combined),
                )
                description = result.response
            except Exception:
                description = combined

            dir_descriptions[dir_uri] = description

            existing = qdrant_service.get_point_by_uri(collection_name=collection_name, uri=dir_uri, point_type="directory")
            point_id = existing.point_id if existing else None
            qdrant_service.upsert_directory_point(
                collection_name=collection_name,
                uri=dir_uri,
                description=description,
                point_id=point_id,
            )
            logger.info("Updated directory point for '%s'", dir_uri)

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

# Global service instance
# In production, this would be dependency-injected
file_service = FileService()
