"""Live test — requires Qdrant and Ollama."""
import json
from pathlib import Path

import pytest

from src.agents.schema_service import (
    discover_schema, infer_facets, get_or_build_schema,
    save_schema, load_schema, _cache_path,
)
from src.agents.models.schema import FacetValue


def test_infer_facets_nss():
    schema = discover_schema("nss")
    facets = infer_facets(schema)

    print("\n--- Inferred facets ---")
    for category, values in facets.items():
        print(f"  {category}: {[fv.value for fv in values]}")

    assert isinstance(facets, dict)
    for category, values in facets.items():
        for fv in values:
            assert isinstance(fv, FacetValue)
            assert fv.source == "inferred"


def test_cache_roundtrip(tmp_path, monkeypatch):
    import src.agents.schema_service as ss
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    schema = discover_schema("nss")
    schema.inferred_facets = infer_facets(schema)
    save_schema("nss", schema)

    assert (tmp_path / "nss.json").exists()

    loaded = load_schema("nss")
    assert loaded is not None
    assert loaded.computed_at == schema.computed_at
    assert loaded.has_images == schema.has_images


def test_get_or_build_uses_cache(tmp_path, monkeypatch):
    import src.agents.schema_service as ss
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    s1 = get_or_build_schema("nss")
    s2 = get_or_build_schema("nss")
    # Second call must return the cached version (same timestamp)
    assert s1.computed_at == s2.computed_at, "Cache not used on second call"
    print("\nCache round-trip OK — computed_at unchanged:", s1.computed_at)
