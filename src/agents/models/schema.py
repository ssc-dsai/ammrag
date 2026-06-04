from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class FacetValue:
    value: str
    source: Literal["inferred", "observed"]

    def to_dict(self) -> dict:
        return {"value": self.value, "source": self.source}

    @staticmethod
    def from_dict(d: dict) -> "FacetValue":
        return FacetValue(value=d["value"], source=d["source"])


@dataclass
class CollectionSchema:
    has_images: bool
    has_structured: bool
    uri_segments: list[str]
    inferred_facets: dict[str, list[FacetValue]]
    directory_summaries: list[str]
    computed_at: str        # ISO 8601 UTC
    collection_version: int # -1 = unknown; matches ingest version counter

    def to_json(self) -> str:
        d = {
            "has_images": self.has_images,
            "has_structured": self.has_structured,
            "uri_segments": self.uri_segments,
            "inferred_facets": {
                k: [fv.to_dict() for fv in vs]
                for k, vs in self.inferred_facets.items()
            },
            "directory_summaries": self.directory_summaries,
            "computed_at": self.computed_at,
            "collection_version": self.collection_version,
        }
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "CollectionSchema":
        d = json.loads(s)
        return CollectionSchema(
            has_images=d["has_images"],
            has_structured=d["has_structured"],
            uri_segments=d["uri_segments"],
            inferred_facets={
                k: [FacetValue.from_dict(fv) for fv in vs]
                for k, vs in d.get("inferred_facets", {}).items()
            },
            directory_summaries=d.get("directory_summaries", []),
            computed_at=d["computed_at"],
            collection_version=d.get("collection_version", -1),
        )
