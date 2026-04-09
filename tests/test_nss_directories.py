"""
Query the 'nss' Qdrant collection for all directory-type points and print as markdown.

Usage:
    python tests/test_nss_directories.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qdrant_client import QdrantClient, models
from src.core.config import settings

COLLECTION = "nss"


def fetch_all_directories(client: QdrantClient) -> list[dict]:
    """Scroll through the entire collection, returning only directory-type points."""
    results = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="directory"),
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
        )
        results.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    return results


def to_markdown(points: list) -> str:
    lines = [
        f"# NSS Directory Points ({len(points)} found)\n",
    ]

    for i, point in enumerate(points, 1):
        payload = point.payload or {}
        uri = payload.get("uri", "")
        text = payload.get("text", "")

        lines.append(f"## {i}. `{uri or point.id}`")
        lines.append(f"- **Point ID:** `{point.id}`")
        if text:
            lines.append(f"- **Description:** {text}")
        lines.append("")

    return "\n".join(lines)


def main():
    kwargs = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    client = QdrantClient(**kwargs)

    print(f"Connecting to Qdrant at {settings.qdrant_url}, collection '{COLLECTION}'...\n")
    points = fetch_all_directories(client)
    print(to_markdown(points))


if __name__ == "__main__":
    main()
