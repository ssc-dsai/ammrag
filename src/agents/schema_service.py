from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote

from src.agents.models.schema import CollectionSchema, FacetValue
from src.models.qdrant_models import FileVector, DirectoryVector
from src.services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".schema_cache"
_NUMERIC_RE = re.compile(r"^\d+$")


# ── cache helpers ────────────────────────────────────────────────────────────

def _cache_path(collection_name: str) -> Path:
    return _CACHE_DIR / f"{collection_name}.json"


def load_schema(collection_name: str) -> CollectionSchema | None:
    p = _cache_path(collection_name)
    if not p.exists():
        return None
    try:
        return CollectionSchema.from_json(p.read_text())
    except Exception as exc:
        logger.warning("Failed to load schema cache for %s: %s", collection_name, exc)
        return None


def save_schema(collection_name: str, schema: CollectionSchema) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(collection_name).write_text(schema.to_json())
    logger.info("Schema cache saved for collection '%s'", collection_name)


# ── discovery ────────────────────────────────────────────────────────────────

def discover_schema(collection_name: str) -> CollectionSchema:
    """Sample the collection and build a CollectionSchema from what is found."""
    logger.info("Discovering schema for collection '%s'", collection_name)

    # Pull up to 50 of each type via broad searches
    broad_queries = ["document", "image", "report", "file", "office"]
    file_vecs: list[FileVector] = []
    dir_vecs: list[DirectoryVector] = []
    seen_ids: set[str] = set()

    for q in broad_queries:
        results = qdrant_service.search(q, collection_name=collection_name, limit=20,
                                        point_type=["file", "directory"])
        for r in results:
            if r.point_id in seen_ids:
                continue
            seen_ids.add(r.point_id)
            if isinstance(r, FileVector):
                file_vecs.append(r)
            elif isinstance(r, DirectoryVector):
                dir_vecs.append(r)

    logger.info("Sampled %d file vectors, %d directory vectors", len(file_vecs), len(dir_vecs))

    # Prevalence flags
    has_images = any(v.payload.image for v in file_vecs)
    has_structured = any(v.payload.structured for v in file_vecs)

    # URI segment vocabulary — decode percent-encoding, split on /
    uri_segments: list[str] = []
    seen_segments: set[str] = set()
    all_vecs = list(file_vecs) + list(dir_vecs)
    for v in all_vecs:
        uri = v.get_uri() or ""
        path = unquote(urlparse(uri).path)
        for seg in path.strip("/").split("/"):
            seg = seg.strip()
            if seg and seg not in seen_segments:
                seen_segments.add(seg)
                uri_segments.append(seg)

    # Directory summaries (capped at 20)
    directory_summaries = [
        (v.get_payload_field("text") or "")[:200]
        for v in dir_vecs[:20]
    ]

    return CollectionSchema(
        has_images=has_images,
        has_structured=has_structured,
        uri_segments=uri_segments,
        inferred_facets={},
        directory_summaries=directory_summaries,
        computed_at=datetime.now(timezone.utc).isoformat(),
        collection_version=-1,
    )


# ── facet inference ───────────────────────────────────────────────────────────

def _has_meaningful_segments(schema: CollectionSchema) -> bool:
    """Return True if URIs have enough structure to warrant LLM facet extraction."""
    meaningful = [s for s in schema.uri_segments if not _NUMERIC_RE.match(s)
                  and len(s) > 2 and "." not in s]
    return len(meaningful) >= 4


def infer_facets(schema: CollectionSchema) -> dict[str, list[FacetValue]]:
    """Extract implicit facet categories from URI segments and directory summaries."""
    if not _has_meaningful_segments(schema):
        logger.info("Heuristic skip: URI structure too flat for facet extraction")
        return {}

    from src.llm.ollama import get_openai_client

    sample_uris = "\n".join(schema.uri_segments[:30])
    sample_summaries = "\n".join(f"- {s}" for s in schema.directory_summaries[:10])

    prompt = (
        "You are analysing a document collection. "
        "Given the URI path segments and directory descriptions below, identify any recurring "
        "categories of meaning (e.g. location names, dates, document types, project names). "
        "Return a JSON object mapping each category name to a list of observed values. "
        "If no clear categories exist, return {}. Return raw JSON only.\n\n"
        f"URI segments:\n{sample_uris}\n\n"
        f"Directory descriptions:\n{sample_summaries}"
    )

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gemma4:12B",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        if "```" in raw:
            raw = raw.split("```")[-2].lstrip("json").strip()
        data: dict = json.loads(raw)
        result: dict[str, list[FacetValue]] = {}
        for category, values in data.items():
            if isinstance(values, list):
                result[category] = [
                    FacetValue(str(v), "inferred") for v in values if v
                ]
        logger.info("Inferred facets: %s", {k: [fv.value for fv in vs] for k, vs in result.items()})
        return result
    except Exception as exc:
        logger.warning("Facet inference failed: %s", exc)
        return {}


# ── main entry point ──────────────────────────────────────────────────────────

def get_or_build_schema(collection_name: str) -> CollectionSchema:
    """Return cached schema if fresh; do partial or full rebuild if stale."""
    from src.agents.ingest_contract import get_collection_version, get_dirty_facets

    cached = load_schema(collection_name)
    live_version = get_collection_version(collection_name)

    if cached is not None and cached.collection_version == live_version:
        logger.info("Using cached schema for '%s' (v%d, computed %s)",
                    collection_name, live_version, cached.computed_at)
        return cached

    if cached is not None and live_version > cached.collection_version:
        dirty = get_dirty_facets(collection_name)
        logger.info("Schema stale for '%s' (cached v%d, live v%d) — rebuilding dirty=%s",
                    collection_name, cached.collection_version, live_version, dirty)

        if dirty == {"facets"}:
            # Partial rebuild: re-run facet inference only
            fresh = discover_schema(collection_name)
            cached.inferred_facets = infer_facets(fresh)
            cached.collection_version = live_version
            save_schema(collection_name, cached)
            return cached

        if dirty == {"has_images"}:
            fresh = discover_schema(collection_name)
            cached.has_images = fresh.has_images
            cached.collection_version = live_version
            save_schema(collection_name, cached)
            return cached

        if dirty == {"has_structured"}:
            fresh = discover_schema(collection_name)
            cached.has_structured = fresh.has_structured
            cached.collection_version = live_version
            save_schema(collection_name, cached)
            return cached

        # Full rebuild for deletions or unknown dirty set
        logger.info("Full schema rebuild for '%s'", collection_name)

    else:
        logger.info("No cached schema for '%s' — building from scratch", collection_name)

    schema = discover_schema(collection_name)
    schema.inferred_facets = infer_facets(schema)
    schema.collection_version = live_version
    save_schema(collection_name, schema)
    return schema
