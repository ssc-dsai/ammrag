"""Live test — requires Qdrant at 192.168.68.92:6333 with 'nss' collection."""
import json
from datetime import datetime

import pytest

from src.agents.schema_service import discover_schema
from src.agents.models.schema import CollectionSchema


def test_discover_schema_nss():
    schema = discover_schema("nss")

    print("\n--- CollectionSchema for 'nss' ---")
    print(f"has_images:      {schema.has_images}")
    print(f"has_structured:  {schema.has_structured}")
    print(f"computed_at:     {schema.computed_at}")
    print(f"collection_ver:  {schema.collection_version}")
    print(f"uri_segments ({len(schema.uri_segments)}): {schema.uri_segments[:10]}")
    print(f"dir summaries ({len(schema.directory_summaries)}): {schema.directory_summaries[:2]}")

    assert isinstance(schema.has_images, bool)
    assert isinstance(schema.has_structured, bool)
    assert len(schema.uri_segments) > 0, "Should have found URI segments"
    # computed_at should parse as ISO 8601
    datetime.fromisoformat(schema.computed_at.replace("Z", "+00:00"))
    assert schema.collection_version == -1  # not yet wired to ingest counter
