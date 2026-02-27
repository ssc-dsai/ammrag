"""
CrewAI service for running agent crews from FastAPI

Calls the QueryRetrievalCrew directly to handle natural language
queries over the indexed document corpus.
"""

import asyncio
import logging
from typing import Dict, Any, List

from fastapi import HTTPException

from src.core.config import settings

logger = logging.getLogger(__name__)


class CrewService:
    """Service for running the query retrieval crew."""

    async def query(
        self,
        question: str,
        collection_names: List[str],
        timeout: int | None = None,
    ) -> Dict[str, Any]:
        """
        Run the QueryRetrievalCrew to answer a natural language question.

        The crew pipeline:
        1. RAG Retriever - vector search across all catalog collections
        2. Structured Data Analyst - detect and query structured tables
        3. Response Compiler - compile Markdown response with source URIs

        Args:
            question: The user's natural language question
            collection_names: List of Qdrant collection names to search
            timeout: Maximum execution time in seconds

        Returns:
            Dict with keys: text, images, files, sources

        Raises:
            HTTPException: If crew execution fails or times out
        """
        timeout = timeout or settings.ollama_timeout

        logger.info("Starting QueryRetrievalCrew for: %s", question[:120])

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._run_query_sync, question, collection_names
                ),
                timeout=timeout,
            )
            logger.info(
                "QueryRetrievalCrew completed, response length=%d chars",
                len(result.get("text", "")),
            )
            return result

        except asyncio.TimeoutError:
            logger.error("QueryRetrievalCrew timed out after %ds", timeout)
            raise HTTPException(
                status_code=504,
                detail=f"Query crew timed out after {timeout}s",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("QueryRetrievalCrew failed")
            raise HTTPException(
                status_code=500,
                detail=f"Query crew error: {str(e)}",
            )

    @staticmethod
    def _run_query_sync(
        question: str, collection_names: List[str]
    ) -> Dict[str, Any]:
        """Synchronous wrapper - imported lazily to avoid heavy crewai
        imports at service startup."""
        from src.agents.retrieval_crew import run_query
        return run_query(question, collection_names)


# Global service instance
crew_service = CrewService()
