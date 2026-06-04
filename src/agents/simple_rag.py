from __future__ import annotations

import logging

from src.agents.models.intent import QueryIntent
from src.services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.3  # hybrid embeddings in this setup score ~0.3-0.5 for good matches


def simple_rag(query: str, collection_name: str, threshold: float = _CONFIDENCE_THRESHOLD) -> str | None:
    """
    Single-pass chunk search. Returns a formatted answer string if the top
    result is confident enough, or None to signal the full pipeline should run.
    """
    logger.info("simple_rag: searching for '%s' in '%s'", query[:60], collection_name)
    # Search all vector types — in image-heavy collections the best text match
    # may be in file-level descriptions rather than chunk vectors.
    results = qdrant_service.search(query, collection_name=collection_name, limit=1)

    if not results:
        logger.info("simple_rag: no results found — falling back to full pipeline")
        return None

    top = results[0]
    score = top.score
    logger.info("simple_rag: top result score=%.3f (threshold=%.3f)", score, threshold)

    if score < threshold:
        logger.info("simple_rag: score below threshold — falling back to full pipeline")
        return None

    text = top.get_payload_field("text") or ""
    uri = top.get_uri() or ""
    logger.info("simple_rag: returning direct answer from %s", uri)
    return f"{text}\n\nSource: {uri}"
