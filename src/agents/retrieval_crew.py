"""
Unified retrieval crew for the query_documents MCP tool.

Pipeline:
1. RAG Retriever       – vector-searches all collections for the catalog
2. Structured Analyst  – detects structured tables in results, builds & runs SQL
3. Response Compiler   – merges everything into a Markdown document with source URIs

IMPORTANT: verbose is disabled because this crew runs inside the MCP server
which uses stdio transport. Any stdout output corrupts the JSON-RPC protocol.
All diagnostic output goes to stderr via logging.
"""

import contextlib
import io
import json
import logging
import os
import sys
import warnings
from typing import List

# Suppress CrewAI tracing/telemetry banners BEFORE importing crewai.
# These print decorative boxes and interactive prompts to stdout which
# corrupts the MCP stdio JSON-RPC transport.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TESTING", "true")

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import NL2SQLTool

from src.agents.tools.fastapi_client import (
    MultiCollectionVectorSearchTool,
    MetadataSearchTool,
)
from src.core.config import settings

logger = logging.getLogger(__name__)

# Suppress noisy LiteLLM/Pydantic serialization warnings (Ollama responses
# often have fewer fields than the OpenAI schema expects).
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings.*",
    category=UserWarning,
)


def _unwrap_stdout() -> None:
    """Restore stdout if CrewAI wrapped it in a FilteredStream."""
    original = getattr(sys.stdout, "_original_stream", None)
    if original is not None:
        sys.stdout = original


def _get_llm() -> LLM:
    """Create LLM instance from environment config."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
    # LiteLLM requires the provider prefix for routing
    if not ollama_model.startswith("ollama/"):
        ollama_model = f"ollama/{ollama_model}"
    return LLM(model=ollama_model, base_url=ollama_host)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr to prevent CrewAI/LiteLLM print()
    calls from corrupting the MCP stdio transport.
    Also suppresses noisy Pydantic serialization warnings from LiteLLM
    when using Ollama (response model has fewer fields than expected)."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
            )
            yield
    finally:
        sys.stdout = old_stdout


@CrewBase
class QueryRetrievalCrew:
    """
    Retrieval crew with three agents:
    - RAG Retriever: searches vector collections for relevant documents
    - Structured Data Analyst: detects structured tables and queries them with SQL
    - Response Compiler: produces a Markdown document with source URIs
    """

    agents_config = "config/query/agents.yaml"
    tasks_config = "config/query/tasks.yaml"

    # ── agents ──────────────────────────────────────────────────────────

    @agent
    def rag_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_retriever"],  # type: ignore[index]
            tools=[MultiCollectionVectorSearchTool()],
            verbose=False,
            llm=_get_llm(),
            max_iter=3,
        )

    @agent
    def structured_data_analyst(self) -> Agent:
        nl2sql = NL2SQLTool(db_uri=settings.postgres_dsn)
        return Agent(
            config=self.agents_config["structured_data_analyst"],  # type: ignore[index]
            tools=[nl2sql, MetadataSearchTool()],
            verbose=False,
            llm=_get_llm(),
            max_iter=5,
        )

    @agent
    def response_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config["response_compiler"],  # type: ignore[index]
            verbose=False,
            llm=_get_llm(),
            max_iter=3,
        )

    # ── tasks ───────────────────────────────────────────────────────────

    @task
    def retrieve_documents(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_documents"],  # type: ignore[index]
        )

    @task
    def analyze_structured_data(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_structured_data"],  # type: ignore[index]
        )

    @task
    def compile_response(self) -> Task:
        return Task(
            config=self.tasks_config["compile_response"],  # type: ignore[index]
        )

    # ── crew ────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Creates the QueryRetrievalCrew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
            tracing=False,
        )


def run_query(question: str, collection_names: List[str]) -> dict:
    """
    Run the retrieval crew and return a structured response.

    This is the public API called directly from the MCP server.
    stdout is redirected to stderr during execution to protect
    the MCP stdio transport.

    Args:
        question: The user's natural language question
        collection_names: List of Qdrant collection names to search

    Returns:
        dict with keys: text, images, files, sources
    """
    logger.info("run_query called: question=%r, collections=%s", question[:80], collection_names)
    _unwrap_stdout()

    try:
        with _redirect_stdout_to_stderr():
            crew_instance = QueryRetrievalCrew()
            result = crew_instance.crew().kickoff(
                inputs={
                    "question": question,
                    "collection_names": json.dumps(collection_names),
                }
            )

        raw_output = str(result)
        logger.info("Crew finished, output length=%d", len(raw_output))

        # Try to parse the crew output as JSON
        try:
            parsed = json.loads(raw_output)
            return {
                "text": parsed.get("text", raw_output),
                "images": parsed.get("images", []),
                "files": parsed.get("files", []),
                "sources": parsed.get("sources", []),
            }
        except json.JSONDecodeError:
            # The response compiler returns markdown directly
            return {
                "text": raw_output,
                "images": [],
                "files": [],
                "sources": [],
            }
    except Exception as e:
        logger.error("QueryRetrievalCrew failed: %s", e, exc_info=True)
        return {
            "text": f"Error processing query: {str(e)}",
            "images": [],
            "files": [],
            "sources": [],
        }
    finally:
        _unwrap_stdout()
