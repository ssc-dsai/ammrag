"""
CrewAI service for running the RAGFlow from FastAPI.
"""

import asyncio
import logging
from typing import Dict, Any

from fastapi import HTTPException

from src.core.config import settings

logger = logging.getLogger(__name__)


class CrewService:
    """Service for running the RAGFlow pipeline."""

    async def query(
        self,
        question: str,
        project_name: str | None = None,
        timeout: int | None = None,
    ) -> Dict[str, Any]:
        timeout = timeout or settings.ollama_timeout * 3

        logger.info("=== RAGFlow query started ===")
        logger.info("Question   : %s", question)
        logger.info("Project    : %s", project_name or "(default)")
        logger.info("Timeout    : %ds", timeout)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_flow_sync, question, project_name),
                timeout=timeout,
            )
            logger.info("=== RAGFlow query completed ===")
            logger.info("Sources: %s", result.get("sources", []))
            return result

        except asyncio.TimeoutError:
            logger.error("RAGFlow timed out after %ds", timeout)
            raise HTTPException(status_code=504, detail=f"RAGFlow timed out after {timeout}s")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("RAGFlow failed: %s", e)
            raise HTTPException(status_code=500, detail=f"RAGFlow error: {e}")

    @staticmethod
    def _run_flow_sync(question: str, collection_name: str | None = None) -> Dict[str, Any]:
        """Synchronous wrapper — imported lazily to avoid heavy imports at startup."""
        from src.agents.flows.rag import RAGFlow
        result = RAGFlow().kickoff(inputs={"query": question, "collection_name": collection_name})
        if isinstance(result, dict):
            return result
        return {"aspects": [], "sources": [], "images": [], "files": []}


# Global service instance
crew_service = CrewService()
