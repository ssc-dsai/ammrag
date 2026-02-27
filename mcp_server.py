"""
MCP Server for File Import Service

This MCP server provides tools for importing files via URIs and retrieving them.
It exposes the FastAPI service functionality through the Model Context Protocol.
"""

import os
import sys

# Suppress CrewAI tracing/telemetry banners BEFORE any crewai import.
# These print decorative boxes and interactive prompts to stdout which
# corrupts the MCP stdio JSON-RPC transport.
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TESTING"] = "true"

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, HttpUrl, ConfigDict
from typing import Optional, List
from enum import Enum
import asyncio
import httpx
import json
import logging

# Ensure project root is on path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging: everything to stderr so stdout stays clean for MCP stdio ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_server")

# Initialize MCP server
mcp = FastMCP("file_import_mcp")

# Base URL for the FastAPI service
API_BASE_URL = "http://localhost:8000"


class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class ImportSingleInput(BaseModel):
    """Input model for single file import."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    uri: HttpUrl = Field(
        ...,
        description="URI of the file to import (e.g., 'https://example.com/document.pdf')"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class ImportBatchInput(BaseModel):
    """Input model for batch file import."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    uris: List[HttpUrl] = Field(
        ...,
        description="List of file URIs to import (e.g., ['https://example.com/doc1.pdf', 'https://example.com/doc2.pdf'])",
        min_length=1,
        max_length=50
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class RetrieveFileInput(BaseModel):
    """Input model for file retrieval."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    file_uuid: str = Field(
        ...,
        description="UUID of the file to retrieve (e.g., '123e4567-e89b-12d3-a456-426614174000')",
        min_length=36,
        max_length=36
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class QueryFilesInput(BaseModel):
    """Input model for querying files."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


def _handle_api_error(e: Exception) -> str:
    """Consistent error formatting across all tools."""
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 404:
            return "Error: Resource not found. Please check the UUID or URI is correct."
        elif e.response.status_code == 422:
            return "Error: Invalid input. Please check your parameters are correctly formatted."
        elif e.response.status_code == 400:
            return "Error: Bad request. Please ensure all required parameters are provided."
        elif e.response.status_code == 500:
            return "Error: Server error. The API service may be experiencing issues."
        return f"Error: API request failed with status {e.response.status_code}"
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. Please try again or check if the API service is running."
    elif isinstance(e, httpx.ConnectError):
        return f"Error: Cannot connect to API service at {API_BASE_URL}. Please ensure the FastAPI service is running."
    return f"Error: Unexpected error occurred: {type(e).__name__}: {str(e)}"


def _format_import_response(data: dict, format: ResponseFormat) -> str:
    """Format import response in markdown or JSON."""
    if format == ResponseFormat.JSON:
        return json.dumps(data, indent=2)
    
    # Markdown format
    if isinstance(data.get("files"), list):
        # Batch import
        output = [f"# Batch Import Complete\n"]
        output.append(f"**Total Files Imported:** {data['total']}\n")
        for i, file in enumerate(data["files"], 1):
            output.append(f"## File {i}")
            output.append(f"- **UUID:** `{file['uuid']}`")
            output.append(f"- **URI:** {file['uri']}")
            output.append(f"- **Imported At:** {file['imported_at']}\n")
    else:
        # Single import
        output = [f"# File Import Complete\n"]
        output.append(f"- **UUID:** `{data['uuid']}`")
        output.append(f"- **URI:** {data['uri']}")
        output.append(f"- **Imported At:** {data['imported_at']}")
    
    return "\n".join(output)


def _format_retrieve_response(data: dict, format: ResponseFormat) -> str:
    """Format retrieve response in markdown or JSON."""
    if format == ResponseFormat.JSON:
        return json.dumps(data, indent=2)
    
    # Markdown format
    output = [f"# File Information\n"]
    output.append(f"- **UUID:** `{data['uuid']}`")
    output.append(f"- **Original URI:** {data['uri']}")
    output.append(f"- **Imported At:** {data['imported_at']}")
    output.append(f"- **Access Link:** {data['access_link']}")
    output.append(f"\n**Download URL:** `{API_BASE_URL}{data['access_link']}`")
    
    return "\n".join(output)


@mcp.tool(
    name="file_import_single",
    annotations={
        "title": "Import Single File",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def import_single_file(params: ImportSingleInput) -> str:
    """Import a single file from a URI and receive a unique UUID.
    
    This tool imports a file by its URI and stores metadata in the system.
    The file is assigned a unique UUID that can be used to retrieve it later.
    
    Args:
        params (ImportSingleInput): Input parameters containing:
            - uri (HttpUrl): URI of the file to import
            - response_format (ResponseFormat): Output format (markdown or json)
    
    Returns:
        str: Formatted response containing the file's UUID, original URI, and import timestamp.
             Format depends on response_format parameter.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/import/single",
                params={"uri": str(params.uri)},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
        return _format_import_response(data, params.response_format)
    
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="file_import_batch",
    annotations={
        "title": "Import Multiple Files",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def import_batch_files(params: ImportBatchInput) -> str:
    """Import multiple files at once from their URIs.
    
    This tool imports multiple files by their URIs in a single operation.
    Each file is assigned a unique UUID that can be used to retrieve it later.
    
    Args:
        params (ImportBatchInput): Input parameters containing:
            - uris (List[HttpUrl]): List of file URIs to import (1-50 files)
            - response_format (ResponseFormat): Output format (markdown or json)
    
    Returns:
        str: Formatted response containing UUIDs and metadata for all imported files.
             Format depends on response_format parameter.
    """
    try:
        # Build query string with multiple uri parameters
        uri_params = [("uri", str(uri)) for uri in params.uris]
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/import/batch",
                params=uri_params,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
        
        return _format_import_response(data, params.response_format)
    
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="file_retrieve",
    annotations={
        "title": "Retrieve File Information",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def retrieve_file(params: RetrieveFileInput) -> str:
    """Retrieve file information and access link using its UUID.
    
    This tool retrieves metadata about a previously imported file using its UUID.
    It returns the original URI, import timestamp, and an access link for downloading.
    
    Args:
        params (RetrieveFileInput): Input parameters containing:
            - file_uuid (str): UUID of the file (36 characters)
            - response_format (ResponseFormat): Output format (markdown or json)
    
    Returns:
        str: Formatted response containing file metadata and download access link.
             Format depends on response_format parameter.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/retrieve/{params.file_uuid}",
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
        
        return _format_retrieve_response(data, params.response_format)
    
    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="file_service_health",
    annotations={
        "title": "Check Service Health",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def check_health(params: QueryFilesInput) -> str:
    """Check the health status of the file import service.
    
    This tool checks if the file import service is running and returns
    information about total files in the system.
    
    Args:
        params (QueryFilesInput): Input parameters containing:
            - response_format (ResponseFormat): Output format (markdown or json)
    
    Returns:
        str: Health status and total file count in the specified format.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/health",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)
        
        # Markdown format
        status_emoji = "✅" if data["status"] == "healthy" else "❌"
        output = [
            f"# Service Health Status\n",
            f"{status_emoji} **Status:** {data['status'].upper()}",
            f"📁 **Total Files:** {data['total_files']}"
        ]
        return "\n".join(output)
    
    except Exception as e:
        return _handle_api_error(e)


# Ollama Tools

class OllamaGenerateInput(BaseModel):
    """Input model for Ollama text generation."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    prompt: str = Field(
        ...,
        description="The prompt to send to the LLM",
        min_length=1
    )
    system: Optional[str] = Field(
        None,
        description="Optional system message to set context"
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 to 2.0)"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum tokens to generate"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class OllamaChatMessage(BaseModel):
    """Single chat message for Ollama"""
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class OllamaChatInput(BaseModel):
    """Input model for Ollama chat completion."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    messages: List[OllamaChatMessage] = Field(
        ...,
        description="List of chat messages with role and content",
        min_length=1
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 to 2.0)"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum tokens to generate"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


@mcp.tool(
    name="ollama_generate",
    annotations={
        "title": "Generate Text with Ollama",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def ollama_generate(params: OllamaGenerateInput) -> str:
    """Generate text using Ollama LLM.

    This tool sends a prompt to Ollama and returns the generated text response.
    You can optionally provide a system message to set the context or personality.

    Args:
        params (OllamaGenerateInput): Input parameters containing:
            - prompt (str): The text prompt to send to the LLM
            - system (Optional[str]): System message for context
            - temperature (float): Sampling temperature (0.0-2.0)
            - max_tokens (Optional[int]): Maximum tokens to generate
            - response_format (ResponseFormat): Output format (markdown or json)

    Returns:
        str: Generated text in the specified format
    """
    try:
        payload = {
            "prompt": params.prompt,
            "temperature": params.temperature
        }

        if params.system:
            payload["system"] = params.system
        if params.max_tokens:
            payload["max_tokens"] = params.max_tokens

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/ollama/generate",
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)

        # Markdown format
        output = [
            f"# Ollama Response\n",
            f"**Model:** {data.get('model', 'unknown')}",
            f"**Prompt:** {params.prompt[:100]}{'...' if len(params.prompt) > 100 else ''}\n",
            f"## Generated Text\n",
            data.get('response', ''),
            f"\n---",
            f"*Tokens: {data.get('eval_count', 'N/A')} | " +
            f"Duration: {round(data.get('total_duration', 0) / 1e9, 2)}s*"
        ]
        return "\n".join(output)

    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="ollama_chat",
    annotations={
        "title": "Chat with Ollama",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def ollama_chat(params: OllamaChatInput) -> str:
    """Have a conversation with Ollama using chat format.

    This tool maintains conversation context by sending message history to Ollama.

    Args:
        params (OllamaChatInput): Input parameters containing:
            - messages (List[OllamaChatMessage]): Conversation history
            - temperature (float): Sampling temperature (0.0-2.0)
            - max_tokens (Optional[int]): Maximum tokens to generate
            - response_format (ResponseFormat): Output format (markdown or json)

    Returns:
        str: Assistant's response in the specified format
    """
    try:
        payload = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in params.messages],
            "temperature": params.temperature
        }

        if params.max_tokens:
            payload["max_tokens"] = params.max_tokens

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/ollama/chat",
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)

        # Markdown format
        message = data.get('message', {})
        output = [
            f"# Chat Response\n",
            f"**Model:** {data.get('model', 'unknown')}",
            f"**Role:** {message.get('role', 'assistant')}\n",
            f"## Message\n",
            message.get('content', ''),
            f"\n---",
            f"*Tokens: {data.get('eval_count', 'N/A')} | " +
            f"Duration: {round(data.get('total_duration', 0) / 1e9, 2)}s*"
        ]
        return "\n".join(output)

    except Exception as e:
        return _handle_api_error(e)


@mcp.tool(
    name="ollama_health",
    annotations={
        "title": "Check Ollama Service",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def check_ollama_health(params: QueryFilesInput) -> str:
    """Check if Ollama service is available and working.

    This tool checks the health of the Ollama integration and shows
    configuration details.

    Args:
        params (QueryFilesInput): Input parameters containing:
            - response_format (ResponseFormat): Output format (markdown or json)

    Returns:
        str: Ollama service status in the specified format
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/ollama/health",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)

        # Markdown format
        status_emoji = "✅" if data["available"] else "❌"
        output = [
            f"# Ollama Service Status\n",
            f"{status_emoji} **Status:** {data['status'].upper()}",
            f"🌐 **Host:** {data['ollama_host']}",
            f"🤖 **Model:** {data['ollama_model']}",
            f"📊 **Available Models:** {data.get('models_count', 'N/A')}"
        ]
        return "\n".join(output)

    except Exception as e:
        return _handle_api_error(e)


# Query Documents Tool (primary RAG interface)

class QueryDocumentsInput(BaseModel):
    """Input model for querying documents with the retrieval crew."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    question: str = Field(
        ...,
        description="Natural language question to answer using the document knowledge base",
        min_length=1,
    )
    name: Optional[str] = Field(
        None,
        description="Catalog name to query. At least one of 'name' or 'catalog_id' must be provided.",
    )
    catalog_id: Optional[int] = Field(
        None,
        description="Catalog ID to query. At least one of 'name' or 'catalog_id' must be provided.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured or 'markdown' for human-readable",
    )


@mcp.tool(
    name="query_documents",
    annotations={
        "title": "Query Documents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def query_documents(params: QueryDocumentsInput) -> str:
    """Query the document knowledge base using AI-powered retrieval.

    This tool analyzes your question, searches across vector and metadata stores,
    and synthesizes a comprehensive answer with supporting references.

    Requires at least one of 'name' or 'catalog_id' to identify which catalog
    to query. The tool resolves the catalog's collections and searches them.

    Args:
        params (QueryDocumentsInput): Input parameters containing:
            - question (str): Natural language question to answer
            - name (Optional[str]): Catalog name to query
            - catalog_id (Optional[int]): Catalog ID to query
            - response_format (ResponseFormat): Output format (json or markdown)

    Returns:
        str: Structured response with text answer, images, files, and sources.
    """
    try:
        if not params.name and params.catalog_id is None:
            return "Error: At least one of 'name' or 'catalog_id' must be provided."

        logger.info(
            "query_documents: question=%r, name=%s, catalog_id=%s",
            params.question[:80], params.name, params.catalog_id,
        )

        # Resolve catalog and its collections
        from src.services.postgres_service import postgres_service
        catalogs = postgres_service.get_catalogs(
            id=params.catalog_id, name=params.name
        )
        if not catalogs:
            logger.warning("query_documents: no catalog found for name=%s id=%s", params.name, params.catalog_id)
            return "Error: No catalog found matching the provided name or catalog_id."

        catalog = catalogs[0]
        logger.info("query_documents: resolved catalog id=%s name=%s", catalog["id"], catalog.get("name"))

        collections = postgres_service.get_collections(catalog_id=catalog["id"])
        if not collections:
            logger.warning("query_documents: no collections for catalog %s", catalog["id"])
            return "Error: No collections found for the specified catalog."

        collection_names = [c["name"] for c in collections]
        logger.info("query_documents: searching collections %s", collection_names)

        # Run the retrieval crew directly (no HTTP round-trip)
        from src.agents.retrieval_crew import run_query
        result = await asyncio.to_thread(
            run_query, params.question, collection_names
        )

        logger.info("query_documents: crew returned, text length=%d", len(result.get("text", "")))

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(result, indent=2)

        # Markdown format — the result is already markdown from the response compiler
        return result.get("text", "No answer generated.")

    except Exception as e:
        logger.error("query_documents failed: %s", e, exc_info=True)
        return _handle_api_error(e)


# Resource to expose service information
@mcp.resource("service://info")
async def get_service_info() -> str:
    """Expose service information as an MCP resource.

    Returns:
        str: JSON containing service endpoints and capabilities
    """
    info = {
        "service": "AMMRAG - File Import, Ollama & CrewAI",
        "version": "1.0.0",
        "api_base_url": API_BASE_URL,
        "capabilities": {
            "import_single": "Import individual files via URI",
            "import_batch": "Import multiple files at once",
            "retrieve": "Retrieve file metadata and access links",
            "health": "Check service status",
            "ollama_generate": "Generate text with Ollama LLM",
            "ollama_chat": "Chat conversations with Ollama",
            "ollama_health": "Check Ollama service availability",
            "query_documents": "AI-powered document Q&A using retrieval agents",
        },
        "endpoints": {
            "import_single": "/import/single",
            "import_batch": "/import/batch",
            "retrieve": "/retrieve/{uuid}",
            "health": "/health",
            "ollama_generate": "/ollama/generate",
            "ollama_chat": "/ollama/chat",
            "ollama_models": "/ollama/models",
            "ollama_health": "/ollama/health",
            "crews_query": "/crews/query",
            "search_query": "/search/query",
            "search_vectors": "/search/vectors",
            "search_metadata": "/search/metadata",
            "search_documents": "/search/documents/{id}",
        }
    }
    return json.dumps(info, indent=2)


if __name__ == "__main__":
    # Run with stdio transport (default for MCP servers)
    mcp.run()
