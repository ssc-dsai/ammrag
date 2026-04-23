"""
CrewAI service for running the RAGFlow from FastAPI.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any

from fastapi import HTTPException

from src.core.config import settings

logger = logging.getLogger(__name__)


class CrewService:
    """Service for running the RAGFlow pipeline."""

    async def query(
        self,
        question: str,
        project_name: str | None = None,
        synthesis: bool = False,
        timeout: int | None = None,
    ) -> Dict[str, Any]:
        timeout = timeout or settings.ollama_timeout * 3

        logger.info("=== RAGFlow query started ===")
        logger.info("Question   : %s", question)
        logger.info("Project    : %s", project_name or "(default)")
        logger.info("Synthesis  : %s", synthesis)
        logger.info("Timeout    : %ds", timeout)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_flow_sync, question, project_name, None, synthesis),
                timeout=timeout,
            )
            logger.info("=== RAGFlow query completed ===")
            logger.info("Sources: %s", result.get("sources", []))
            logger.info("=== Final answer ===\n%s", result.get("answer", "(no answer)"))
            return result

        except asyncio.TimeoutError:
            logger.error("RAGFlow timed out after %ds", timeout)
            raise HTTPException(status_code=504, detail=f"RAGFlow timed out after {timeout}s")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("RAGFlow failed: %s", e)
            raise HTTPException(status_code=500, detail=f"RAGFlow error: {e}")

    async def stream_query(
        self,
        question: str,
        project_name: str | None = None,
        synthesis: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the RAGFlow pipeline, yielding a progress event at each stage
        and a final result event when complete."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(stage: str, n: int, total: int) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "progress", "stage": stage, "n": n, "total": total},
            )

        def run() -> None:
            try:
                result = self._run_flow_sync(question, project_name, on_progress, synthesis)
            except Exception as e:
                logger.exception("stream_query flow error: %s", e)
                result = {"answer": f"Error: {e}", "sources": [], "images": [], "files": []}
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", **result})

        loop.run_in_executor(None, run)

        while True:
            event = await queue.get()
            logger.info("stream_query event: %s", event.get("type") if event.get("type") == "result" else event)
            yield event
            if event["type"] == "result":
                break

    @staticmethod
    def _run_flow_sync(
        question: str,
        collection_name: str | None = None,
        on_progress=None,
        synthesis: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper — imported lazily to avoid heavy imports at startup."""
        from src.agents.flows.rag import RAGFlow
        inputs = {"query": question, "collection_name": collection_name, "synthesis": synthesis}
        if on_progress is not None:
            inputs["on_progress"] = on_progress
        result = RAGFlow().kickoff(inputs=inputs)
        if isinstance(result, dict):
            return result
        return {"answer": "", "sources": [], "images": [], "files": []}


# Global service instance
crew_service = CrewService()
