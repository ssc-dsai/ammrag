"""
Custom CrewAI tools that make HTTP calls to FastAPI search endpoints.
Used by the QueryRetrievalCrew agents to access data through the API layer.
"""

import logging
import httpx
from typing import Any, Type, Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


FASTAPI_BASE_URL = "http://localhost:8000"

logger = logging.getLogger(__name__)


def _http_error_detail(e: Exception) -> str:
    """Extract useful detail from an HTTP error."""
    if isinstance(e, httpx.HTTPStatusError):
        try:
            # FastAPI returns {"detail": "..."} for HTTPException
            import json as _json
            body = e.response.text[:1000]
            try:
                parsed = _json.loads(body)
                body = parsed.get("detail", body)
            except _json.JSONDecodeError:
                pass
        except Exception:
            body = "(could not read body)"
        return f"HTTP {e.response.status_code}: {body}"
    if isinstance(e, httpx.ConnectError):
        return f"Connection refused — is the FastAPI server running at {FASTAPI_BASE_URL}?"
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Multi-collection vector search
# ---------------------------------------------------------------------------

class MultiCollectionVectorSearchInput(BaseModel):
    query: str = Field(description="The semantic search query text")
    collection_names: List[str] = Field(description="List of Qdrant collection names to search")
    limit: int = Field(default=10, description="Maximum number of results per collection")
    score_threshold: float = Field(
        default=0.0, description="Minimum similarity score (0.0-1.0)"
    )


class MultiCollectionVectorSearchTool(BaseTool):
    """Search for documents across multiple Qdrant collections using semantic similarity."""

    name: str = "vector_search"
    description: str = (
        "Search for documents across multiple Qdrant collections using semantic similarity. "
        "Provide a natural language query and a list of collection names. "
        "Returns relevant text chunks ranked by similarity score from all collections."
    )
    args_schema: Type[BaseModel] = MultiCollectionVectorSearchInput

    def _run(
        self,
        query: str,
        collection_names: List[str],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> Any:
        all_results = []
        for collection_name in collection_names:
            payload = {
                "query": query,
                "collection_name": collection_name,
                "limit": limit,
                "score_threshold": score_threshold,
            }
            try:
                logger.info("vector_search: querying collection %s", collection_name)
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        f"{FASTAPI_BASE_URL}/search/vectors", json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
                    results = data.get("results", [])
                    for r in results:
                        r["collection_name"] = collection_name
                        # Promote key fields from nested metadata to top level
                        # so the agent can easily see URIs and table indicators
                        meta = r.get("metadata") or {}
                        if "uri" not in r and "uri" in meta:
                            r["uri"] = meta["uri"]
                        node = meta.get("node") or {}
                        if "uri" not in r and isinstance(node, dict) and "uri" in node:
                            r["uri"] = node["uri"]
                        if "pg_table_name" not in r and "pg_table_name" in meta:
                            r["pg_table_name"] = meta["pg_table_name"]
                    all_results.extend(results)
                    logger.info("vector_search: got %d results from %s", len(results), collection_name)
            except Exception as e:
                detail = _http_error_detail(e)
                logger.error("vector_search failed for %s: %s", collection_name, detail)
                all_results.append({"error": detail, "collection_name": collection_name})

        # Sort by score descending
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results


# ---------------------------------------------------------------------------
# SQL query tool for structured data tables
# ---------------------------------------------------------------------------

class SQLQueryInput(BaseModel):
    table_name: str = Field(description="The PostgreSQL table name to query")
    sql_query: str = Field(
        default="",
        description=(
            "A SQL SELECT query. Example: SELECT * FROM my_table LIMIT 10. "
            "If left empty, defaults to SELECT * FROM <table_name> LIMIT 20. "
            "Only SELECT statements are allowed."
        ),
    )


class SQLQueryTool(BaseTool):
    """Execute a read-only SQL query against a PostgreSQL structured data table."""

    name: str = "sql_query"
    description: str = (
        "Execute a read-only SELECT SQL query against a structured data table in PostgreSQL. "
        "Use this to query tables discovered in vector search results (via the 'pg_table_name' field). "
        "Provide the table_name. You can optionally provide a sql_query; if omitted, "
        "it defaults to SELECT * FROM <table_name> LIMIT 20. Returns columns and rows."
    )
    args_schema: Type[BaseModel] = SQLQueryInput

    def _run(self, table_name: str, sql_query: str = "") -> Any:
        # Auto-generate query if the agent didn't provide one
        if not sql_query or not sql_query.strip():
            sql_query = f"SELECT * FROM {table_name} LIMIT 20"
            logger.info("sql_query: auto-generated query for table=%s", table_name)

        logger.info("sql_query: table=%s query=%s", table_name, sql_query[:200])
        try:
            payload = {"sql_query": sql_query}
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{FASTAPI_BASE_URL}/search/sql", json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            detail = _http_error_detail(e)
            logger.error("sql_query failed: %s", detail)

            # On failure, try to return sample data so the agent can see
            # actual column names and retry with a corrected query.
            fallback = self._sample_table(table_name)
            if fallback and "error" not in fallback:
                return {
                    "error": detail,
                    "hint": "The query failed. Here are the actual columns and sample rows for this table.",
                    "columns": fallback.get("columns", []),
                    "sample_rows": fallback.get("rows", [])[:5],
                }
            return {"error": detail}

    def _sample_table(self, table_name: str) -> Any:
        """Fetch sample rows to reveal column names on query failure."""
        try:
            fallback_sql = f"SELECT * FROM {table_name} LIMIT 5"
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{FASTAPI_BASE_URL}/search/sql",
                    json={"sql_query": fallback_sql},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Single-collection vector search (kept for backward compatibility)
# ---------------------------------------------------------------------------

class VectorSearchInput(BaseModel):
    query: str = Field(description="The semantic search query text")
    limit: int = Field(default=10, description="Maximum number of results")
    score_threshold: float = Field(
        default=0.0, description="Minimum similarity score (0.0-1.0)"
    )
    collection_name: Optional[str] = Field(
        default=None, description="Qdrant collection name (uses default if not set)"
    )


class VectorSearchTool(BaseTool):
    """Search for documents using semantic similarity via FastAPI."""

    name: str = "vector_search_single"
    description: str = (
        "Search for documents in a single collection using semantic similarity. "
        "Provide a natural language query and get back relevant text chunks "
        "ranked by similarity score."
    )
    args_schema: Type[BaseModel] = VectorSearchInput

    def _run(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        collection_name: Optional[str] = None,
    ) -> Any:
        payload = {
            "query": query,
            "limit": limit,
            "score_threshold": score_threshold,
        }
        if collection_name:
            payload["collection_name"] = collection_name

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{FASTAPI_BASE_URL}/search/vectors", json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            detail = _http_error_detail(e)
            logger.error("vector_search_single failed: %s", detail)
            return {"error": detail}


# ---------------------------------------------------------------------------
# Metadata search
# ---------------------------------------------------------------------------

class MetadataSearchInput(BaseModel):
    query: Optional[str] = Field(
        default=None, description="Text to search in metadata"
    )
    table_name: Optional[str] = Field(
        default=None, description="Specific table to query"
    )
    limit: int = Field(default=50, description="Maximum number of results")


class MetadataSearchTool(BaseTool):
    """Search PostgreSQL metadata tables via FastAPI."""

    name: str = "metadata_search"
    description: str = (
        "Search PostgreSQL metadata tables for structured information. "
        "Can query specific tables or search across metadata. "
        "Useful for finding table names, document metadata, and structured data."
    )
    args_schema: Type[BaseModel] = MetadataSearchInput

    def _run(
        self,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        limit: int = 50,
    ) -> Any:
        payload: dict = {"limit": limit}
        if query:
            payload["query"] = query
        if table_name:
            payload["table_name"] = table_name

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{FASTAPI_BASE_URL}/search/metadata", json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            detail = _http_error_detail(e)
            logger.error("metadata_search failed: %s", detail)
            return {"error": detail}


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------

class DocumentRetrievalInput(BaseModel):
    document_id: str = Field(
        description="The ID of the document to retrieve from the vector store"
    )


class DocumentRetrievalTool(BaseTool):
    """Retrieve a specific document by ID via FastAPI."""

    name: str = "get_document"
    description: str = (
        "Retrieve a specific document by its ID from the vector store. "
        "Returns the full document content and metadata."
    )
    args_schema: Type[BaseModel] = DocumentRetrievalInput

    def _run(self, document_id: str) -> Any:
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{FASTAPI_BASE_URL}/search/documents/{document_id}"
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            detail = _http_error_detail(e)
            logger.error("get_document failed: %s", detail)
            return {"error": detail}
