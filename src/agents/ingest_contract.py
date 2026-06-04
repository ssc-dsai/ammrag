from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agents.schema_service import _CACHE_DIR

logger = logging.getLogger(__name__)


def _version_path(collection_name: str) -> Path:
    return _CACHE_DIR / f"{collection_name}_version.json"


def get_collection_version(collection_name: str) -> int:
    """Return the current ingest version counter for a collection (-1 if unknown)."""
    p = _version_path(collection_name)
    if not p.exists():
        return -1
    try:
        data = json.loads(p.read_text())
        return int(data.get("version", -1))
    except Exception:
        return -1


def get_dirty_facets(collection_name: str) -> set[str]:
    """Return the set of dirty facet categories from the last ingest run."""
    p = _version_path(collection_name)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        return set(data.get("dirty_facets", []))
    except Exception:
        return set()


def increment_collection_version(
    collection_name: str,
    dirty_facets: set[str],
) -> int:
    """
    Increment the version counter and record which facet categories changed.
    Called by the ingest pipeline after modifying the collection.
    Returns the new version number.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    current = get_collection_version(collection_name)
    new_version = current + 1
    data = {
        "version": new_version,
        "dirty_facets": sorted(dirty_facets),
    }
    _version_path(collection_name).write_text(json.dumps(data, indent=2))
    logger.info(
        "Collection '%s' version incremented to %d (dirty: %s)",
        collection_name, new_version, dirty_facets,
    )
    return new_version
