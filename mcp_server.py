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

from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field
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
    ctx: Context,
    question: str = Field(..., description="Natural language question to answer", min_length=1),
    synthesis: bool = Field(default=False, description="If true, synthesise a formatted markdown answer from the analysis results. If false, return the raw data points."),
) -> str:
    """Query the document knowledge base using the RAGFlow pipeline.

    Steps: query decomposition → vector search → analysis → optional answer synthesis.
    When synthesis=True, returns a markdown-formatted answer with inline citation links.
    When synthesis=False, returns the raw analysis data points.
    """
    try:
        payload: dict = {"question": question, "synthesis": synthesis}
        if settings.mcp_collection_name:
            payload["project_name"] = settings.mcp_collection_name

        timeout = httpx.Timeout(connect=10.0, read=settings.ollama_timeout * 3 + 30.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{API_BASE_URL}/crews/query/stream", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "progress":
                        await ctx.report_progress(event["n"], event["total"], event["stage"])
                    elif event["type"] == "result":
                        return format_crew_response(event)

        return "No response received."

    except Exception as e:
        logger.error("query_documents failed: %s", e, exc_info=True)
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
            "query_documents": "POST /crews/query/stream — RAGFlow pipeline with optional synthesis",
        },
    }
    return json.dumps(info, indent=2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
