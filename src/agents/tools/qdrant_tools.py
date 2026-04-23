import json
import logging
from typing import Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from src.services.qdrant_service import qdrant_service
from src.models.qdrant_models import QdrantVector
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1 – Qdrant Search
# ---------------------------------------------------------------------------

class _SearchInput(BaseModel):
    query: str = Field(description="Natural-language query to search the vector database for")
    limit: int = Field(default=5, description="Maximum number of chunks to return (default 5)")


class QdrantSearchTool(BaseTool):
    """Search Qdrant for document chunks relevant to a query."""

    name: str = "qdrant_search"
    description: str = (
        "Search the Qdrant vector database for document chunks relevant to a query string. "
        "Returns a ranked list of chunks, each with a point_id, relevance score, source document, "
        "chunk_index, and text excerpt. Always use this tool first when looking for information."
    )
    args_schema: Type[BaseModel] = _SearchInput
    collection_name: Optional[str] = None

    def _run(self, query: str, limit: int = 5) -> List[QdrantVector]:
        try:
            return qdrant_service.search(query, collection_name=self.collection_name, limit=limit)
        except Exception as exc:
            logger.error("QdrantSearchTool failed (query=%r): %s", query[:80], exc)
            raise Exception(f"Error in QdrantSearchTool: {exc}") from exc


# ---------------------------------------------------------------------------
# Tool 2 – Get Neighboring Chunks
# ---------------------------------------------------------------------------

class _NeighborsInput(BaseModel):
    chunk_id: str = Field(
        description="The chunk_id of the chunk whose neighbors you want to retrieve"
    )
    chunk_total: str = Field(
        description="The chunk_total of the chunk whose neighbors you want to retrieve"
    )
    file_uuid: str = Field(
        description="The file_uuid of the chunk whose neighbors you want to retrieve"
    )
    window: int = Field(
        default=2,
        description="How many chunks before AND after the target chunk to fetch (default 2)",
    )


class QdrantGetNeighborsTool(BaseTool):
    """Retrieve the chunks immediately before and after a given chunk."""

    name: str = "qdrant_get_neighbors"
    description: str = (
        "Retrieve the chunks that appear immediately before and after a specific chunk in its source document. "
        "Use this to expand context around a relevant chunk without fetching the whole document. "
        "Requires a chunk_id, chunk_total, and file_uuid returned by qdrant_search."
    )
    args_schema: Type[BaseModel] = _NeighborsInput

    def _run(self, chunk_id: str, chunk_total: str, file_uuid: str, window: int = 2) -> str:
        try:
            chunk_start = max(int(chunk_id) - window, 0)
            chunk_end = min(int(chunk_id) + window, int(chunk_total) - 1)
            return qdrant_service.fetch_chunks_as_text(file_uuid, chunk_start, chunk_end)
        except Exception as exc:
            logger.error("QdrantGetNeighborsTool failed (file_uuid=%s, chunk_id=%s): %s", file_uuid, chunk_id, exc)
            return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool 3 – Get Full Document
# ---------------------------------------------------------------------------

class _DocumentInput(BaseModel):
    file_uuid: str = Field(
        description="The file_uuid of the document to retrieve in full, as returned by qdrant_search"
    )
    chunk_total: str = Field(
        description="The chunk_total for this document, as returned by qdrant_search"
    )


class QdrantGetDocumentTool(BaseTool):
    """Retrieve every chunk that belongs to a document identified by file_uuid."""

    name: str = "qdrant_get_document"
    description: str = (
        "Retrieve the complete text of a source document. "
        "Use this when neighboring chunks are still not sufficient to answer a question. "
        "Requires the file_uuid returned by qdrant_search."
    )
    args_schema: Type[BaseModel] = _DocumentInput

    def _run(self, file_uuid: str, chunk_total: str) -> str:
        try:
            return qdrant_service.fetch_chunks_as_text(file_uuid, chunk_start=0, chunk_end=int(chunk_total) - 1)
        except Exception as exc:
            logger.error("QdrantGetDocumentTool failed (file_uuid=%s): %s", file_uuid, exc)
            return json.dumps({"error": str(exc)})
