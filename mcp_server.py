"""
MCP Server for AMMRAG

Exposes the FastAPI service functionality through the Model Context Protocol.
Each tool corresponds directly to a FastAPI endpoint.
"""

import os
import sys

# Suppress CrewAI tracing/telemetry banners BEFORE any crewai import.
# These print decorative boxes and interactive prompts to stdout which
# corrupts the MCP stdio JSON-RPC transport.
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TESTING"] = "true"

# Ensure project root is on path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Optional, List, Dict, Any
from enum import Enum
import httpx
import json
import logging

from src.core.logging_config import setup_logging
from src.utils.response_formatter import format_crew_response

setup_logging()
logger = logging.getLogger("mcp_server")

from src.core.config import settings

API_BASE_URL = f"http://{settings.host}:{settings.port}"

mcp = FastMCP("ammrag_mcp", host="0.0.0.0", port=settings.mcp_port)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def _handle_api_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 404:
            return "Error: Resource not found."
        elif e.response.status_code == 422:
            return f"Error: Invalid input — {e.response.text}"
        elif e.response.status_code == 400:
            return "Error: Bad request. Check your parameters."
        elif e.response.status_code == 500:
            return "Error: Server error."
        return f"Error: API request failed with status {e.response.status_code}"
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out."
    elif isinstance(e, httpx.ConnectError):
        return f"Error: Cannot connect to API at {API_BASE_URL}. Is the FastAPI service running?"
    return f"Error: {type(e).__name__}: {e}"





# ---------------------------------------------------------------------------
# Crews (RAG)
# ---------------------------------------------------------------------------

@mcp.tool(
    name="query_documents",
    annotations={"title": "Query Documents", "readOnlyHint": True, "openWorldHint": True},
)
async def query_documents(
    question: str = Field(..., description="Natural language question to answer", min_length=1),
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for structured",
    ),
) -> str:
    """Query the document knowledge base using the RAGFlow pipeline.

    Steps: query decomposition → vector search → context widening → answer synthesis.
    """
    try:
        async with httpx.AsyncClient() as client:
            payload: dict = {"question": question}
            if settings.mcp_collection_name:
                payload["project_name"] = settings.mcp_collection_name
            r = await client.post(
                f"{API_BASE_URL}/crews/query",
                json=payload,
                timeout=settings.ollama_timeout + 30.0,
            )
            r.raise_for_status()
            data = r.json()

        if response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)

        return format_crew_response(data)

    except Exception as e:
        logger.error("query_documents failed: %s", e, exc_info=True)
        return _handle_api_error(e)




# ---------------------------------------------------------------------------
# File tree (directory + file descriptions)
# ---------------------------------------------------------------------------

@mcp.tool(
    name="file_tree_descriptions",
    annotations={"title": "File Tree Descriptions", "readOnlyHint": True, "openWorldHint": False},
)
async def file_tree_descriptions(
    question: str = Field(..., description="Natural language question used to find relevant files and directories", min_length=1),
    limit: int = Field(default=5, description="Maximum number of results per point type (directories and files each)", ge=1, le=20),
) -> str:
    """Search for relevant directory and file description points in the knowledge base.

    Returns summaries of matching directories and files to help narrow down
    which areas of the document tree are relevant before calling query_documents.
    """
    try:
        from src.services.qdrant_service import qdrant_service

        collection = settings.mcp_collection_name or settings.qdrant_collection_name

        dirs = qdrant_service.search(question, collection_name=collection, limit=limit, point_type="directory")
        files = qdrant_service.search(question, collection_name=collection, limit=limit, point_type="file")

        lines: list[str] = []

        if dirs:
            lines.append("## Directories\n")
            for v in dirs:
                uri = v.get_payload_field("uri") or "unknown"
                text = (v.get_payload_field("text") or "").strip()
                lines.append(f"- **{uri}**\n  {text}\n")

        if files:
            lines.append("## Files\n")
            for v in files:
                uri = v.get_payload_field("uri") or "unknown"
                text = (v.get_payload_field("text") or "").strip()
                lines.append(f"- **{uri}**\n  {text}\n")

        if not lines:
            return "No matching directories or files found."

        return "\n".join(lines)

    except Exception as e:
        logger.error("file_tree_descriptions failed: %s", e, exc_info=True)
        return _handle_api_error(e)


# ---------------------------------------------------------------------------
# Service info resource
# ---------------------------------------------------------------------------

@mcp.resource("service://info")
async def get_service_info() -> str:
    """Expose service information and available tool→endpoint mapping."""
    info = {
        "service": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "api_base_url": API_BASE_URL,
        "tools": {
            "file_tree_descriptions": "qdrant direct (directory + file points)",
            "query_documents":        "POST /crews/query (chunk points only)",
        },
    }
    return json.dumps(info, indent=2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
