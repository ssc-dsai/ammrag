"""Tests for multi-hop re-planning and schema write-back."""
import json
from pathlib import Path

import pytest

from src.agents.models.analysis import DataPoint
from src.agents.models.schema import CollectionSchema, FacetValue
from src.agents.replanner import replan_from_discovery


def make_schema() -> CollectionSchema:
    return CollectionSchema(
        has_images=True,
        has_structured=False,
        uri_segments=[],
        inferred_facets={},
        directory_summaries=[],
        computed_at="2026-04-29T00:00:00Z",
        collection_version=-1,
    )


def make_datapoints(quotes: list[str]) -> list[DataPoint]:
    return [
        DataPoint(uri="test://uri", type="text", pertinent=True,
                  step="discovery", quote=q, reasoning="discovery result")
        for q in quotes
    ]


def test_replan_generates_info_needs(tmp_path, monkeypatch):
    import src.agents.schema_service as ss
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    schema = make_schema()
    dps = make_datapoints([
        "The offices are located at 100 Saint Joseph Road and 101 Boul Roland-Therien.",
        "There are two office buildings in the collection.",
    ])

    needs = replan_from_discovery(
        original_query="Compare kitchens across all offices",
        discovery_datapoints=dps,
        schema=schema,
        collection_name="nss",
    )

    print("\n--- Generated InfoNeeds ---")
    for n in needs:
        print(f"  info={n.info!r}  query={n.query!r}")

    assert len(needs) >= 1, "Expected at least one InfoNeed from entity discovery"
    for n in needs:
        assert n.info
        assert n.query


def test_replan_writes_back_to_schema_cache(tmp_path, monkeypatch):
    import src.agents.schema_service as ss
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    schema = make_schema()
    dps = make_datapoints([
        "Offices: 100 Saint Joseph Road and 101 Boul Roland-Therien."
    ])

    replan_from_discovery(
        original_query="Compare kitchens across all offices",
        discovery_datapoints=dps,
        schema=schema,
        collection_name="nss",
    )

    # Schema cache should have been written
    cache_file = tmp_path / "nss.json"
    assert cache_file.exists(), "Schema cache not written after replan"

    data = json.loads(cache_file.read_text())
    facets = data.get("inferred_facets", {})
    print(f"\n--- Written facets ---")
    print(json.dumps(facets, indent=2))

    # At least one observed entity should be present
    all_facet_values = [fv for vs in facets.values() for fv in vs]
    observed = [fv for fv in all_facet_values if fv.get("source") == "observed"]
    assert len(observed) >= 1, f"Expected observed entities in cache, got: {facets}"


def test_replan_subsequent_call_finds_cached_entities(tmp_path, monkeypatch):
    import src.agents.schema_service as ss
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path)

    schema = make_schema()
    dps = make_datapoints(["Offices: Alpha Building and Beta Campus."])

    replan_from_discovery("Compare offices", dps, schema, "nss")

    # Load the schema back and verify entities are present
    loaded = ss.load_schema("nss")
    assert loaded is not None
    all_values = [fv.value for vs in loaded.inferred_facets.values() for fv in vs]
    print(f"\n--- Cached entity values ---: {all_values}")
    assert len(all_values) >= 1
