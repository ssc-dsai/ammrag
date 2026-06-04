import json
import pytest
from src.agents.models.schema import FacetValue, CollectionSchema


def make_schema() -> CollectionSchema:
    return CollectionSchema(
        has_images=True,
        has_structured=False,
        uri_segments=["offices", "london", "floorplans"],
        inferred_facets={
            "location": [
                FacetValue("london", "inferred"),
                FacetValue("sydney", "observed"),
            ]
        },
        directory_summaries=["Architectural drawings for the London office"],
        computed_at="2026-04-29T12:00:00Z",
        collection_version=3,
    )


def test_facetvalue_source_valid():
    fv = FacetValue("london", "inferred")
    assert fv.source == "inferred"
    fv2 = FacetValue("sydney", "observed")
    assert fv2.source == "observed"


def test_schema_json_roundtrip():
    schema = make_schema()
    serialised = schema.to_json()
    d = json.loads(serialised)

    assert d["has_images"] is True
    assert d["has_structured"] is False
    assert "london" in d["uri_segments"]
    assert d["collection_version"] == 3

    restored = CollectionSchema.from_json(serialised)
    assert restored.has_images == schema.has_images
    assert restored.collection_version == schema.collection_version
    assert restored.computed_at == schema.computed_at
    assert len(restored.inferred_facets["location"]) == 2


def test_facetvalue_roundtrip():
    fv = FacetValue("london", "inferred")
    restored = FacetValue.from_dict(fv.to_dict())
    assert restored.value == "london"
    assert restored.source == "inferred"


def test_empty_facets_roundtrip():
    schema = CollectionSchema(
        has_images=False,
        has_structured=False,
        uri_segments=[],
        inferred_facets={},
        directory_summaries=[],
        computed_at="2026-04-29T00:00:00Z",
        collection_version=-1,
    )
    restored = CollectionSchema.from_json(schema.to_json())
    assert restored.inferred_facets == {}
    assert restored.collection_version == -1
