import os
import json
from functools import lru_cache
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition,MatchAny, MatchValue, Range


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def _get_collection() -> str:
    return os.environ.get("QDRANT_COLLECTION", "documents")


@lru_cache(maxsize=4)
def _get_vector_name(collection: str) -> Optional[str]:
    """Return the named vector to query against.

    Priority:
    1. QDRANT_VECTOR_NAME env var (explicit override)
    2. Auto-detect: if the collection has exactly one named vector, use it
    3. None  — falls back to the collection's default/unnamed vector
    """
    explicit = os.environ.get("QDRANT_VECTOR_NAME")
    if explicit:
        return explicit
    try:
        info = _get_client().get_collection(collection)
        vectors = info.config.params.vectors
        # Named-vector collections: vectors is a dict of name -> config
        if isinstance(vectors, dict) and vectors:
            return next(iter(vectors))
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    """Load and cache the SentenceTransformer model (loaded once per process)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _get_embedding(text: str) -> list[float]:
    """Generate a vector embedding using a local SentenceTransformers model."""
    model_name = os.environ.get("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
    model = _get_model(model_name)
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def _parse_point_id(point_id: str):
    """Convert a string point ID back to int or keep as UUID string."""
    try:
        return int(point_id)
    except ValueError:
        return point_id
    

def _text(payload: dict) -> str:
    return str(payload.get("chunk", ""))


def _source(payload: dict) -> str:
    return str(payload.get("uri", "unknown"))


def _chunk_index(payload: dict) -> Optional[int]:
    val = payload.get("chunk_id")
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass
    return None


def _source_key(_payload: dict) -> Optional[str]:
    return "uri"


def _fetch_chunks_as_text(file_id: str, chunk_start: int, chunk_end: int) -> str:
    """Scroll Qdrant for chunks of file_id in [chunk_start, chunk_end], sort by chunk_id,
    join their text, and return as a JSON string {"text": "..."}."""
    client = _get_client()
    collection = _get_collection()
    limit = chunk_end - chunk_start + 1

    all_chunks, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="file_id", match=MatchValue(value=file_id)),
                FieldCondition(key="chunk_id", range=Range(gte=chunk_start, lte=chunk_end)),
            ]
        ),
        limit=limit,
        with_payload=True,
    )

    sorted_chunks = sorted(
        all_chunks,
        key=lambda p: (_chunk_index(p.payload or {}) or 0),
    )
    combined_text = " ".join(_text(p.payload or {}) for p in sorted_chunks)
    return json.dumps({"text": combined_text}, indent=2)



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

    def _run(self, query: str, limit: int = 5) -> str:
        try:
            embedding = _get_embedding(query)
            client = _get_client()
            collection = _get_collection()
            results = client.query_points(
                collection_name=collection,
                query=embedding,
                using=_get_vector_name(collection),
                limit=limit,
                with_payload=True,
            )
            if not results.points:
                return json.dumps({"results": [], "message": "No results found."})

            output = []
            for rank, r in enumerate(results.points, start=1):
                payload = r.payload or {}
                output.append({
                    "rank": rank,
                    "point_id": str(r.id),
                    "score": round(r.score, 4),
                    "uri": payload.get("uri"),
                    "location": payload.get("location", ""),
                    "uuid": payload.get("uuid", ""),
                    "chunk_id": payload.get("chunk_id"),
                    "chunk_total": payload.get("chunk_total"),
                    "file_id": payload.get("file_id"),
                    "text": payload.get("chunk"),
                })
            return json.dumps({"results": output}, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})



class _NeighborsInput(BaseModel):
    chunk_id: str = Field(
        description="The chunk_id of the chunk whose neighbors you want to retrieve"
    )
    chunk_total: str = Field(
        description="The chunk_total of the chunk whose neighbors you want to retrieve"
    )
    file_id: str = Field(
        description="The file_id of the chunk whose neighbors you want to retrievek"
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
        "Requires a chunk_id, chunk_total, and file_id returned by qdrant_search."
    )
    args_schema: Type[BaseModel] = _NeighborsInput

    def _run(self, chunk_id: str, chunk_total: str, file_id: str, window: int = 2) -> str:
        try:
            chunk_start = max(int(chunk_id) - window, 0)
            chunk_end = min(int(chunk_id) + window, int(chunk_total) - 1)
            return _fetch_chunks_as_text(file_id, chunk_start, chunk_end)
        except Exception as exc:
            return json.dumps({"error": str(exc)})


class _DocumentInput(BaseModel):
    file_id: str = Field(
        description="The file_id of the document to retrieve in full, as returned by qdrant_search"
    )
    chunk_total: str = Field(
        description="The chunk_total for this document, as returned by qdrant_search"
    )


class QdrantGetDocumentTool(BaseTool):
    """Retrieve every chunk that belongs to a document identified by file_id."""

    name: str = "qdrant_get_document"
    description: str = (
        "Retrieve the complete text of a source document. "
        "Use this when neighboring chunks are still not sufficient to answer a question. "
        "Requires the file_id returned by qdrant_search."
    )
    args_schema: Type[BaseModel] = _DocumentInput

    def _run(self, file_id: str, chunk_total: str) -> str:
        try:
            return _fetch_chunks_as_text(file_id, chunk_start=0, chunk_end=int(chunk_total) - 1)
        except Exception as exc:
            return json.dumps({"error": str(exc)})
