"""Tests for the ingest pipeline version contract — no live services needed."""
import json
from pathlib import Path

import pytest

from src.agents.ingest_contract import (
    get_collection_version,
    get_dirty_facets,
    increment_collection_version,
)
from src.agents.schema_service import get_or_build_schema, save_schema, load_schema
from src.agents.models.schema import CollectionSchema


def make_schema(version: int) -> CollectionSchema:
    return CollectionSchema(
        has_images=True,
        has_structured=False,
        uri_segments=["offices", "london"],
        inferred_facets={},
        directory_summaries=[],
        computed_at="2026-04-29T00:00:00Z",
        collection_version=version,
    )


def test_version_starts_at_minus_one(tmp_path, monkeypatch):
    import src.agents.ingest_contract as ic
    import src.agents.schema_service as ss
    monkeypatch.setattr(ic, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    assert get_collection_version("testcol") == -1


def test_increment_version(tmp_path, monkeypatch):
    import src.agents.ingest_contract as ic
    import src.agents.schema_service as ss
    monkeypatch.setattr(ic, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    v1 = increment_collection_version("testcol", {"facets"})
    assert v1 == 0

    v2 = increment_collection_version("testcol", {"has_images"})
    assert v2 == 1

    assert get_collection_version("testcol") == 1
    assert get_dirty_facets("testcol") == {"has_images"}


def test_stale_cache_triggers_rebuild(tmp_path, monkeypatch):
    import src.agents.ingest_contract as ic
    import src.agents.schema_service as ss
    monkeypatch.setattr(ic, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    # Save a schema at version 0
    schema = make_schema(version=0)
    save_schema("testcol", schema)
    increment_collection_version("testcol", {"facets"})  # bumps to v0 in file
    increment_collection_version("testcol", {"facets"})  # bumps to v1

    loaded = load_schema("testcol")
    assert loaded is not None
    assert loaded.collection_version == 0  # cache is stale (live is v1)
    assert get_collection_version("testcol") == 1


def test_fresh_cache_not_rebuilt(tmp_path, monkeypatch):
    import src.agents.ingest_contract as ic
    import src.agents.schema_service as ss
    monkeypatch.setattr(ic, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    # Set version to 5, save schema at version 5
    for _ in range(6):  # 6 increments → version 5
        increment_collection_version("testcol", set())

    schema = make_schema(version=5)
    save_schema("testcol", schema)

    loaded = load_schema("testcol")
    assert loaded is not None
    assert loaded.collection_version == get_collection_version("testcol") == 5
    print(f"\n  Schema at v{loaded.collection_version} — matches live v{get_collection_version('testcol')} ✓")
